## 제한 시간을 갖는 작업 환경에서 강화학습 기반 드론의 자율비행 비교 연구

 본 논문은 혈액 운송과 같은 제한 시간을 갖는 작업 환경에서 드론의 자율비행에 관한 것으로, 기존 Proportional-Integral-Derivative (PID) 방식에 비해 강화학습(Reinforcement Learning; RL)을 이용하는 경우의 성능 향상을 비교 실험을 통해서 제시한다. 오픈소스 드론 시뮬레이터를 통해 PID 기반 드론의 자율비행보다 RL 기반 드론의 평균 비행시간이 약 2.8초 빠른 것을 확인할 수 있었다. 이러한 결과를 바탕으로 짧은 시간 내에 목표 달성이 필요로 하는 작업 환경에서 강화학습이 효과적으로 사용될 수 있을 것으로 기대한다.

저자: 권은주,정희철,김현석 저자 소속: 동아대학교 키워드: 강화학습, 드론, PID  한국통신학회
---
## Installation

```sh
git clone https://github.com/eunjuyummy/gym-pybullet-drones.git
sudo apt install ffmpeg

cd gym-pybullet-drones/
sudo pip install -e.

cd gym_pybullet_drones/examples/

python main.py
```
# gym-pybullet-drones

This is a minimalist refactoring of the original `gym-pybullet-drones` repository, designed for compatibility with [`gymnasium`](https://github.com/Farama-Foundation/Gymnasium), [`stable-baselines3` 2.0](https://github.com/DLR-RM/stable-baselines3/pull/1327), and SITL [`betaflight`](https://github.com/betaflight/betaflight)/[`crazyflie-firmware`](https://github.com/bitcraze/crazyflie-firmware/).

> **NOTE**: if you prefer to access the original codebase, presented at IROS in 2021, please `git checkout [paper|master]` after cloning the repo, and refer to the corresponding `README.md`'s.

<img src="files/readme_images/helix.gif" alt="formation flight" width="350"> <img src="files/readme_images/helix.png" alt="control info" width="450">
