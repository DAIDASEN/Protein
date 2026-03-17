"""
Main MCTD-ME algorithm.

Implements Algorithm 1 (Lead Optimization) and Algorithm 2 (De-Novo design)
from Appendix A.1 of arXiv:2509.15796.

Algorithm 1 – Lead Optimization:
  root ← initial fully-denoised lead sequence y_0
  for i = 1 to T:
      # Selection
      node ← root
      while node.children ≠ ∅:
          node ← PH-UCT-ME(node.children)
      # Expansion
      M ← GetMaskSet(node.sequence)   (pLDDT-guided progressive masking)
      Y ← {}
      for each expert e in E:
          for r = 1 to R:
              y_0 ← MaskedDiffusion(node.sequence, M, e)
              if y_0 ∉ cache:
                  score ← EvalComposite(y_0; C)
                  cache[y_0] ← score
              Y ← Y ∪ {y_0}
      Y_top ← TopK(Y, K; cache)
      for y' in Y_top:
          (U_ent(y'), U_div(y')) ← ComputeBonuses(y', node.sequence, M)
          AddChild(node, y', U_ent(y'), U_div(y'))
      # Backpropagation
      for y' in Y_top:
          v ← cache[y']
          Backpropagate(y', v)
  return Top-k sequences by cache[·]

Algorithm 2 – De-Novo:
  Same as Algorithm 1 but root = all-masked sequence.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from mctd_me.config import MCTDMEConfig
from mctd_me.critics import CompositeReward, ESMFoldCritic
from mctd_me.diffusion import MultiExpertRolloutEngine
from mctd_me.experts import BaseExpert, build_experts
from mctd_me.masking import get_mask_set_for_node
from mctd_me.selection import select_child_ph_uct_me
from mctd_me.tree import MCTSNode
from mctd_me.utils import SequenceCache, set_seed, validate_sequence

logger = logging.getLogger(__name__)


class MCTDME:
    """
    Monte Carlo Tree Diffusion with Multiple Experts (MCTD-ME).

    Parameters
    ----------
    config : MCTDMEConfig
        Full algorithm configuration.
    experts : List[BaseExpert] | None
        Pre-built expert instances.  If None, they are constructed from
        config.experts using build_experts().
    critic : CompositeReward | None
        Pre-built composite reward critic.  If None, one is constructed from
        the ESMFold critic specified in config.
    """

    def __init__(
        self,
        config: MCTDMEConfig,
        experts: Optional[List[BaseExpert]] = None,
        critic: Optional[CompositeReward] = None,
        esmfold_critic: Optional[ESMFoldCritic] = None,
    ) -> None:
        self.config = config
        set_seed(config.seed)

        # ---- Experts ----
        if experts is not None:
            self.experts = experts
        else:
            logger.info("Building expert models from config …")
            self.experts = build_experts(
                expert_names=config.experts,
                use_proteinmpnn=config.use_proteinmpnn,
                proteinmpnn_path=config.proteinmpnn_path,
                device=config.device,
            )

        # ---- ESMFold critic ----
        if esmfold_critic is not None:
            self.esmfold_critic = esmfold_critic
        else:
            self.esmfold_critic = ESMFoldCritic(
                model_name=config.esmfold_model,
                device=config.device,
            )

        # ---- Composite reward ----
        if critic is not None:
            self.critic = critic
        else:
            self.critic = CompositeReward(
                task=config.task,
                critic=self.esmfold_critic,
                reward_weights_folding=config.reward_weights_folding,
                reward_weights_inv=config.reward_weights_inv,
                reward_weights_motif=config.reward_weights_motif,
                motif_rmsd_cutoff=config.motif_rmsd_hard_cutoff,
                motif_sctm_cutoff=config.motif_sctm_hard_cutoff,
                folding_plddt_bonus=config.reward_folding_plddt_bonus,
            )

        # ---- Rollout engine ----
        self.rollout_engine = MultiExpertRolloutEngine(
            experts=self.experts,
            num_rollouts=config.num_rollouts,
            num_diffusion_steps=config.diffusion_steps,
            temperature=config.temperature,
            deterministic=True,  # argmax decoding (Table 10)
            w_ent=config.w_ent,
            w_div=config.w_div,
        )

        # ---- Sequence cache ----
        self.cache = SequenceCache()

        # ---- Statistics ----
        self._n_evaluations: int = 0
        self._n_cache_hits: int = 0

    # ------------------------------------------------------------------
    # Critic evaluation with caching
    # ------------------------------------------------------------------

    def _evaluate(
        self,
        sequence: str,
        critic_kwargs: dict,
    ) -> Tuple[float, dict]:
        """
        Evaluate *sequence* via the composite reward, with caching.

        Parameters
        ----------
        sequence : str
        critic_kwargs : dict
            Keyword arguments forwarded to self.critic().

        Returns
        -------
        reward : float
        info   : dict
        """
        if sequence in self.cache:
            self._n_cache_hits += 1
            return self.cache[sequence]

        self._n_evaluations += 1
        reward, info = self.critic(sequence, **critic_kwargs)
        self.cache[sequence] = (reward, info)
        return reward, info

    # ------------------------------------------------------------------
    # pLDDT retrieval for masking
    # ------------------------------------------------------------------

    def _get_plddt(self, sequence: str) -> List[float]:
        """
        Return per-residue pLDDT scores (in [0, 1]) for *sequence*.

        Uses the ESMFold critic; results are cached via self.cache.
        """
        if sequence in self.cache:
            _, info = self.cache[sequence]
            if "plddt_per_residue" in info:
                return info["plddt_per_residue"]

        # Run ESMFold just for pLDDT (cheaper than full reward)
        result = self.esmfold_critic.predict(sequence)
        plddt_normalised = [v / 100.0 for v in result["plddt"]]
        return plddt_normalised

    # ------------------------------------------------------------------
    # Selection phase
    # ------------------------------------------------------------------

    def _select(self, root: MCTSNode) -> MCTSNode:
        """
        Descend the tree using PH-UCT-ME until a leaf or unexpanded node.

        Parameters
        ----------
        root : MCTSNode

        Returns
        -------
        MCTSNode
            The selected node for expansion.
        """
        node = root
        while node.children and node.is_expanded:
            # Build consensus prior map for children
            pi_cons_map = {
                child.sequence: getattr(child, "_pi_cons", 1.0)
                for child in node.children
            }
            node = select_child_ph_uct_me(
                parent=node,
                exploration_constant=self.config.exploration_constant,
                w_ent=self.config.w_ent,
                w_div=self.config.w_div,
                pi_cons_map=pi_cons_map,
            )
            # Stop if we reach maximum depth
            if node.depth >= self.config.max_depth:
                break
        return node

    # ------------------------------------------------------------------
    # Expansion phase
    # ------------------------------------------------------------------

    def _expand(
        self,
        node: MCTSNode,
        critic_kwargs: dict,
        structure: Optional[object] = None,
    ) -> List[MCTSNode]:
        """
        Expand *node* by running multi-expert masked diffusion rollouts.

        Returns the list of newly added child nodes.

        Parameters
        ----------
        node : MCTSNode
        critic_kwargs : dict
            Passed to the critic for reward evaluation.
        structure : optional
            Backbone coordinates for structure-conditioned experts.

        Returns
        -------
        List[MCTSNode]
            Top-K newly created children.
        """
        # 1. Get mask set (pLDDT-guided progressive masking)
        plddt = self._get_plddt(node.sequence)
        mask_indices, threshold = get_mask_set_for_node(
            sequence=node.sequence,
            plddt_scores=plddt,
            depth=node.depth,
            max_depth=self.config.max_depth,
            threshold_init=self.config.plddt_mask_threshold,
            threshold_min=self.config.plddt_threshold_min,
            progressive=self.config.progressive_mask,
        )
        logger.debug(
            f"  Expanding node depth={node.depth}: "
            f"|M|={len(mask_indices)}, pLDDT threshold={threshold:.3f}"
        )

        # 2. Multi-expert rollouts
        candidates = self.rollout_engine.expand(
            sequence=node.sequence,
            mask_indices=mask_indices,
            structure=structure,
            parent_sequence=node.sequence,
        )

        # 3. Evaluate all unique candidates
        evaluated: List[Tuple[str, float, float, float]] = []  # (seq, reward, u_ent, u_div)
        for cand in candidates:
            seq = cand["sequence"]
            reward, _ = self._evaluate(seq, critic_kwargs)
            evaluated.append((seq, reward, cand["u_ent"], cand["u_div"], cand.get("pi_cons", 1.0)))

        # 4. Top-K by reward
        evaluated.sort(key=lambda x: x[1], reverse=True)
        top_k = evaluated[: self.config.top_k_children]

        # 5. Add children
        new_children: List[MCTSNode] = []
        for seq, reward, u_ent, u_div, pi_cons in top_k:
            child = node.add_child(
                sequence=seq,
                reward=reward,
                u_ent=u_ent,
                u_div=u_div,
            )
            child._pi_cons = pi_cons  # cache for selection
            new_children.append(child)

        node.is_expanded = True
        return new_children

    # ------------------------------------------------------------------
    # Backpropagation phase
    # ------------------------------------------------------------------

    def _backpropagate(self, children: List[MCTSNode]) -> None:
        """
        Backpropagate rewards from the newly added children.

        Parameters
        ----------
        children : List[MCTSNode]
        """
        for child in children:
            child.backpropagate(child.reward, rule=self.config.backup_rule)

    # ------------------------------------------------------------------
    # Main MCTS loop (Algorithm 1)
    # ------------------------------------------------------------------

    def run(
        self,
        initial_sequence: str,
        critic_kwargs: Optional[dict] = None,
        structure: Optional[object] = None,
        return_top_k: int = 10,
        verbose: bool = True,
    ) -> List[Tuple[str, float, dict]]:
        """
        Run the MCTD-ME algorithm (Algorithm 1 – Lead Optimization / De-Novo).

        Parameters
        ----------
        initial_sequence : str
            Starting sequence.  Use all-'X' or all-masked string for de-novo.
        critic_kwargs : dict | None
            Task-specific keyword arguments for the composite reward:
              - folding:          {'target_coords': np.ndarray}
              - inverse_folding:  {'native_sequence': str, 'target_coords': np.ndarray}
              - motif_scaffolding:{'motif_indices': List[int],
                                   'motif_coords_ref': np.ndarray,
                                   'motif_seq_ref': str}
        structure : optional
            Backbone coordinates for structure-conditioned experts (ProteinMPNN).
        return_top_k : int
            Number of best sequences to return.
        verbose : bool
            Print iteration progress.

        Returns
        -------
        List of (sequence, reward, info_dict), sorted by reward descending.
        """
        if critic_kwargs is None:
            critic_kwargs = {}

        # Evaluate and cache the root sequence
        root_reward, root_info = self._evaluate(initial_sequence, critic_kwargs)

        # Build root node
        root = MCTSNode(
            sequence=initial_sequence,
            depth=0,
            reward=root_reward,
        )

        if verbose:
            logger.info(
                f"MCTD-ME start | task={self.config.task} | "
                f"root_reward={root_reward:.4f} | "
                f"seq_len={len(initial_sequence)} | "
                f"T={self.config.num_mcts_iterations}"
            )

        start_time = time.time()

        for iteration in range(1, self.config.num_mcts_iterations + 1):
            iter_start = time.time()

            # --- Selection ---
            node = self._select(root)

            # Skip if at max depth
            if node.depth >= self.config.max_depth:
                if verbose:
                    logger.debug(f"  [iter {iteration}] node at max depth, skipping")
                continue

            # --- Expansion ---
            new_children = self._expand(
                node, critic_kwargs=critic_kwargs, structure=structure
            )

            # --- Backpropagation ---
            self._backpropagate(new_children)

            iter_time = time.time() - iter_start
            if verbose and (iteration % 10 == 0 or iteration <= 5):
                best_reward = max(
                    (r for _, (r, _) in self.cache.items()), default=0.0
                )
                logger.info(
                    f"  [iter {iteration:4d}/{self.config.num_mcts_iterations}] "
                    f"tree_size={root.subtree_size():4d} | "
                    f"best_reward={best_reward:.4f} | "
                    f"cache_size={len(self.cache):4d} | "
                    f"iter_time={iter_time:.1f}s"
                )

        elapsed = time.time() - start_time
        if verbose:
            logger.info(
                f"MCTD-ME finished in {elapsed:.1f}s | "
                f"evaluations={self._n_evaluations} | "
                f"cache_hits={self._n_cache_hits}"
            )

        # Return top-k sequences from cache
        top_k_results = self.cache.top_k(return_top_k)
        return top_k_results

    # ------------------------------------------------------------------
    # De-Novo design (Algorithm 2)
    # ------------------------------------------------------------------

    def run_denovo(
        self,
        sequence_length: int,
        critic_kwargs: Optional[dict] = None,
        structure: Optional[object] = None,
        return_top_k: int = 10,
        verbose: bool = True,
    ) -> List[Tuple[str, float, dict]]:
        """
        Run de-novo protein design by starting from an all-'A' sequence.

        In de-novo mode the initial sequence contains no prior information;
        all positions are masked at the first expansion step.

        Parameters
        ----------
        sequence_length : int
            Desired length of the designed sequence.
        critic_kwargs : dict | None
            As for run().
        structure : optional
        return_top_k : int
        verbose : bool

        Returns
        -------
        List of (sequence, reward, info_dict), sorted by reward descending.
        """
        if sequence_length <= 0:
            raise ValueError("sequence_length must be > 0")
        # Start with a uniform placeholder sequence (all-Ala)
        initial_seq = "A" * sequence_length
        if verbose:
            logger.info(f"De-novo design: length={sequence_length}")
        return self.run(
            initial_sequence=initial_seq,
            critic_kwargs=critic_kwargs,
            structure=structure,
            return_top_k=return_top_k,
            verbose=verbose,
        )

    # ------------------------------------------------------------------
    # Convenience: lead optimization (Alg. 1 starting from existing lead)
    # ------------------------------------------------------------------

    def run_lead_optimization(
        self,
        lead_sequence: str,
        critic_kwargs: Optional[dict] = None,
        structure: Optional[object] = None,
        return_top_k: int = 10,
        verbose: bool = True,
    ) -> List[Tuple[str, float, dict]]:
        """
        Lead optimization starting from an existing protein sequence.

        This is the primary use case described in Algorithm 1.

        Parameters
        ----------
        lead_sequence : str
            Initial fully-denoised protein sequence.
        critic_kwargs : dict | None
        structure : optional
        return_top_k : int
        verbose : bool

        Returns
        -------
        List of (sequence, reward, info_dict).
        """
        if not validate_sequence(lead_sequence):
            logger.warning(
                "Lead sequence contains non-standard amino acids. "
                "Proceeding with available characters."
            )
        return self.run(
            initial_sequence=lead_sequence,
            critic_kwargs=critic_kwargs,
            structure=structure,
            return_top_k=return_top_k,
            verbose=verbose,
        )
