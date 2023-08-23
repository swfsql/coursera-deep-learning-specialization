### Deep Learning Specialization on Coursera (offered by deeplearning.ai)

Completed: 
- C1W1~3

This repo is the progress of me learning Machine Learning thanks to the video lectures made by Andrew Ng and some other rsources.  
This is a fork from [this](https://github.com/amanchadha/coursera-deep-learning-specialization/tree/d968708a5318457acdea8f61d6acd4d1db86833f). The parent repo contains more information, references and content, so if you are studying ML I recommend checking it out.

This present repo is somewhat a rewrite from the Python jupyter code into Rust, experimenting with `dfdx`, and later `candle` and `burn` ML frameworks. Each file informs what it's related to from the original/parent/python repo.

For the datasets (~1.2GB .zip file + another 550MB download from `setup.sh`) see [amanchadha/coursera-deep-learning-specialization#10](https://github.com/amanchadha/coursera-deep-learning-specialization/issues/10#issuecomment-1085118606).


### Testing

```bash
# to test using cuda (dfdx, candle) and wpgu (burn), run:
cargo test --features="cuda,wpgu"

# to test using the cpu only, run:
cargo test
# note: I dev with the cuda feature, and sometimes I try (and update fixes) for the cpu-only.
```

### Dev

I'm using vscode's devcontainer extension, from which some configurations for dev are pre-defined.  
TODO: WIP: I haven't re-tested (rebuilt) it.

### References

Courses video lectures:
- C1 - [Neural Networks and Deep Learning](https://www.youtube.com/watch?v=CS4cs9xVecg&list=PLkDaE6sCZn6Ec-XTbcX1uRg2_u4xOEky0).
- C2 - [Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization](https://www.youtube.com/watch?v=1waHlpKiNyY&list=PLkDaE6sCZn6Hn0vK8co82zjQtt3T2Nkqc).
- C3 - [Structuring Machine Learning Projects](https://www.youtube.com/watch?v=dFX8k1kXhOw&list=PLkDaE6sCZn6E7jZ9sN_xHwSHOdjUxUW_b).
- C4 - [Convolutional Neural Networks](https://www.youtube.com/watch?v=ArPaAX_PhIs&list=PLkDaE6sCZn6Gl29AoE31iwdVwSG-KnDzF).
- C5 - [Sequence Models](https://www.youtube.com/watch?v=_i3aqgKVNQI&list=PLkDaE6sCZn6F6wUI9tvS_Gw1vaFAx6rd6&pp=iAQB). 
