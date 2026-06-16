# Orkhon model zoo

Named, dated, self-contained model archives. Each folder has weights, tokenizer, samples, eval, a code snapshot, and `run.sh` when those artifacts are stored locally. Imported models may store only a source repo and re-import command. The codenames are explained in [`docs/lineage.md`](../docs/lineage.md).

| Date | Name | Kind | Params | Metric | Folder |
|------|------|------|--------|--------|--------|
| 20260615 | **bengu** | base | 57M | ppl 47.102 | `bengu-20260615/` |
| 20260615 | **bengu-gokturk** | instruct | 57M |  | `bengu-gokturk-20260615/` |
| 20260614 | **bumin** | instruct | 4M | ppl 5.128 | `bumin-20260614/` |
| 20260614 | **istemi** | base | 51M | ppl 46.506 | `istemi-20260614/` |
| 20260614 | **kashgari** | imported | 135M |  | `kashgari-20260614/` |
| 20260614 | **kultigin** | instruct | 22M |  | `kultigin-20260614/` |
| 20260614 | **tonyukuk** | base | 22M | ppl 4.769 | `tonyukuk-20260614/` |
