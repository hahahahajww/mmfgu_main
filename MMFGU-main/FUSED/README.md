# FUSED

This folder contains a standalone implementation of the FUSED idea from the paper:
`Unlearning through Knowledge Overwriting: Reversible Federated Unlearning via Selective Sparse Adapter`.

Current implementation target:
- Client-level federated unlearning in the existing `MMFGU` codebase.
- Reuses the repo's data loading, client split, model, and evaluation pipeline.
- Keeps all new code inside `FUSED/`.

Implemented stages:
1. Standard federated pretraining.
2. CLI: one extra local update per client to rank sensitive parameter tensors by weighted L1 drift.
3. Sparse adapter unlearning: freeze the pretrained base model, train sparse adapters only on retained clients, and merge adapters back into the base model.
4. Reversible export: save base model, adapter state, mask, and merged final model separately.

Run example:

```bash
python FUSED/run_fused.py --data-dir E:\MMFGU\datasets\ele-fashion --num-clients 10 --forget-client-id 0 --federated-rounds 20 --fused-rounds 5 --cli-topk-layers 4 --adapter-density 0.05 --output-dir FUSED/outputs/exp1
```

Important notes:
- This implementation is aligned to the current repo's client-unlearning setup, not the paper's image classification benchmark code.
- The CLI step operates on parameter tensors in `state_dict`, which is the most direct way to adapt the method to the existing model.
- Reversibility is achieved by saving `base_global_model.pt` plus `adapter_state.pt`; removing adapters restores the pretrained model.
