# Before making this repository public

The code is already under the MIT license in `LICENSE`. Visibility stays private until this review is done.

1. Re-scan `git log -p` for absolute machine paths, webhook URLs, and unpublished data.
2. Confirm Release `v0.1.0` contains only the five Job 6663 `no_pos_emb` files used for manuscript Figures 2, 4, and 5, named `model_released_fold1_es.pt` through `model_released_fold5_es.pt`. A private Release becomes public with the repository.
3. Change visibility:

```bash
gh repo edit --visibility public --accept-visibility-change-consequences
```
