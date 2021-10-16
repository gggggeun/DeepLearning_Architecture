# Transformer : Attention is all you need

## Abstract

sequence data를 처리하기 위해 이전까지 많이 쓰이던 model은 recurrent model 또는 CNN(encoder, decoder) model 이었습니다.
또한, 가장 성능이 좋은 모델은 attention 메카니즘을 통한 인코더와 디코더를 연결한 모델였습니다.
본 논문에서는 attention 메커니즘에만 기반하여 recurrence와 convolutions를 사용하지않는 새로운 간단한 네트워크 아키텍처인 Transformer를 제안합니다.

(1) Recurrent model
- symbol position에 따라서 계산
- token 정보 정렬 후 hidden states h_t 업데이트 하면서 계산하게 됨
- 연속적으로 이어지는 것이기 때문에, 병렬 처리도 힘듦 + 긴 문장에서는 그만큼 NN의 input에 넣어줘야 해서 메모리적 한계가 있음

(2) Attention
- 시퀀스 모델링과 변형 모델에서 중요한 역할
- 입력이나 출력 sequence의 거리에 상관 없이 단어간 관계를 연결하는 모델링 가능
- 출력 정보 중 어떤 정보가 중요한지 가중치 부여 → h_t 에 곱해서 결과 계산

(3) Transformer
- recurrence를 제거하고 Attention에 전적으로 의지해서 입력과 출력간 관계 인식
- Attention-> 한 번의 행렬 곱으로 위치 정보가 포함된 시퀀스 한 번에 계산 ⇒ 병렬 처리 가능
