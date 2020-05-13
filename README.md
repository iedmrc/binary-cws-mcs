<h1 align="center">Comparison of Randomized Solutions for
Constrained Vehicle Routing Problem</h1>
<p>
  <img alt="Version" src="https://img.shields.io/badge/version-1.0-blue.svg?cacheSeconds=2592000" />
  <a href="https://github.com/iedmrc/binary-cws-mcs" target="_blank">
    <img alt="Documentation" src="https://img.shields.io/badge/documentation-yes-brightgreen.svg" />
  </a>
  <a href="LICENSE" target="_blank">
    <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg" />
  </a>
</p>

> An implementation of the "[Comparison of Randomized Solutions for
Constrained Vehicle Routing Problem](https://arxiv.org/abs/2005.05871)" paper.

## Install
To install, just clone the repository and install the dependencies via:

```sh
git clone https://github.com/iedmrc/binary-cws-mcs
```
```sh
pip3 install -r requirements.txt
```

## Usage
To run a sample solver:

```sh
python3 main.py
```

### Pseudorandom Number Generator

[prng.py](prng.py) contains these pseudorandom number generators and generates pseudorandom numbers on the fly, as much as needed:

[Lehmers](https://en.wikipedia.org/wiki/Lehmer_random_number_generator):
- LEHMER_0
- LEHMER_1
- LEHMER_2
- LEHMER_3
- LEHMER_4
- LEHMER_5

[Congruential Generators](https://en.wikipedia.org/wiki/Linear_congruential_generator):

- MIXED_CG
- ICG
- EICG

MRG

### Solver Class

Solver class needs to be initialized with a problem path (`problem_path`) and a prng type (`prng_type`). Problem path must be a type of [tsplib95](https://tsplib95.readthedocs.io/).

Solver class has two solver methods. One is **Binary-CWS** and the other one is **Binary-CWS-MCS**. You can call either of these on solver instance, directly. E.g.:

```
vrp_set_path = './Vrp-Set-E'
problem_path = os.path.join(vrp_set_path, 'E-n30-k3.vrp')

solver = Solver(problem_path)

cost, route = solver.binary_cws_mcs(n=50, distributed=True)
```

Monte Carlo simulations of *Binary-CWS-MCS* can be distributed to different cores (processors) easily by setting `distributed=TRUE` with the help of [ray](https://github.com/ray-project/ray).

## Known Issues
- `_process` method of `Solver` performs very slow. This may cause *Binary-CWS-MCS* to take not an affordable time.

## Authors

👤 **Oğuz YAYLA**
* Website: http://www.mat.hacettepe.edu.tr/people/academic/oguz-yayla/index.html
* Twitter: [@oguzyayla](https://twitter.com/oguzyayla)
* Github: [@oguz-yayla](https://github.com/oguz-yayla)

👤 **Şaziye Ece ÖZDEMİR**

👤 **İbrahim Ethem DEMİRCİ**

* Website: https://ibrahim.demirci.com
* Twitter: [@iedmrc](https://twitter.com/iedmrc)
* Github: [@iedmrc](https://github.com/iedmrc)


## Contribution

Please see [CONTRIBUTING](CONTRIBUTING.md) file.

## 📝 License

Copyright © 2020 [Oğuz YAYLA](http://www.mat.hacettepe.edu.tr/people/academic/oguz-yayla/index.html), [Şaziye Ece ÖZDEMİR](), [İbrahim Ethem DEMİRCİ](https://ibrahim.demirci.com).<br />
This project is [MIT](LICENSE) licensed.

Libraries have their own licences. Please check their page for more details.

Problem Sets:
Christofides, N., & Eilon, S. (1969). An Algorithm for the Vehicle-Dispatching Problem. OR, 20(3), 309-318. doi:10.2307/3008733
***