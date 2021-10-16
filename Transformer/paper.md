# Transformer : Attention is all you need
https://arxiv.org/pdf/1706.03762.pdf (논문)
<br>
<br>
## Abstract

sequence data를 처리하기 위해 이전까지 많이 쓰이던 model은 recurrent model 또는 CNN(encoder, decoder) model 이었습니다.
또한, 가장 성능이 좋은 모델은 attention 메카니즘을 통한 인코더와 디코더를 연결한 모델였습니다.
본 논문에서는 attention 메커니즘에만 기반하여 recurrence와 convolutions를 사용하지않는 새로운 간단한 네트워크 아키텍처인 Transformer를 제안합니다. 
두 가지 기계 번역 작업에 대한 실험은 이러한 모델이 __병렬화(Parallelization) 가능__ 하고 훈련에 훨씬 __적은 시간이 소요__ 되는 동시에 __품질이 우수함__ 을 보여줍니다.


(1) Recurrent model
- symbol position에 따라서 계산
- token 정보 정렬 후 hidden states h_t 업데이트 하면서 계산하게 됨
- 연속적으로 이어지는 것이기 때문에, 병렬 처리도 힘듦 + 긴 문장에서는 그만큼 NN의 input에 넣어줘야 해서 메모리적 한계가 있음

![image](https://user-images.githubusercontent.com/74692845/137569135-9d6c78d9-4fd4-4a64-9faf-8ef7e2a9a0b6.png)


[문제상황]
- 하나의 문맥 벡터가 소스 문장의 모든 정보를 가지고 있어야 하므로 성능이 저하됨

[해결방안]
- 그렇다면 매번 소스 문장에서의 출력 전부를 입력으로 받는다면?
- 최신 GPU는 많은 메모리와 빠른 병렬 처리를 지원함

<br>
<br>

(2) Attention
- 시퀀스 모델링과 변형 모델에서 중요한 역할
- 입력이나 출력 sequence의 거리에 상관 없이 단어간 관계를 연결하는 모델링 가능
- 출력 정보 중 어떤 정보가 중요한지 가중치 부여 → h_t 에 곱해서 결과 계산
![image](https://user-images.githubusercontent.com/74692845/137569252-e9cc5dab-4e00-4afd-80bb-4bbc3ce6ceea.png)

<br>
<br>

(3) Transformer
- recurrence를 제거하고 Attention에 전적으로 의지해서 입력과 출력간 관계 인식
- Attention-> 한 번의 행렬 곱으로 위치 정보가 포함된 시퀀스 한 번에 계산 ⇒ 병렬 처리 가능

<br>
<br>

## Background
1. Sequential computation
sequence to sequence한 문제를 푸는 과정에서, Encoder-Decoder 구조의 RNN 모델들이 좋은 성능을 냈다.

2. Long term dependency
RNN의 경우, Long term dependency의 문제가 항상 따라다니고, CNN의 경우 kernel 안에서 O(1)이나, kernel 간 정보가 공유되지 않는다.

<br>
<br>

## Model Architecture
+)이전 seq2seq 구조에서는 인코더와 디코더에서 각각 하나의 RNN이 t개의 시점(time-step)을 가지는 구조. 여기는 인코더와 디코더라는 단위가 N개로 구성되는 구조


<br>
<br>

### Encoder and Decoder structure
![image](https://user-images.githubusercontent.com/74692845/137569796-bbe127a9-0c50-48a3-850f-42ea2c11612c.png)
encoder는 input sequence (x_1,...,x_n)에 대해 다른 representation인 z=(z_1,...,z_n)으 바꿔줍니다. decoder는 z를 받아, output sequence (y1,...,yn)를 하나씩 만들어냅니다.
각각의 step에서 다음 symbol을 만들 때 이전에 만들어진 output(symbol)을 이용합니다. 예를 들어, “저는 사람입니다.”라는 문장에서 ‘사람입니다’를 만들 때, ‘저는’이라는 symbol을 이용하는 거죠. 이런 특성을 auto-regressive 하다고 합니다.


### Encoder and Decoder stacks
- Transformer는 크게 Encoder와 Decoder로 구분된다. 부수적인 다른 구성 요소들이 있으나, Encoder와 Decoder가 가장 핵심이다. Encoder는 아래 그림에서 좌측, Decoder는 우측을 의미한다.
![image](https://user-images.githubusercontent.com/74692845/137569648-af29dea6-7e1a-4da6-b6c1-d70251374ab5.png)


#### 1) Encoder

- Embedding : RNN을 사용하지 않으려면 위치 정보를 포함하고 있는 임베딩을 사용해야한다 (Positional Encoding) (=input)

![image](https://user-images.githubusercontent.com/74692845/137570321-2af067d1-1134-4f8c-8210-f4d8f1b3955e.png)

- Self Attention : 임베딩이 끝난 후 어텐션(Attention)진행. 각각의 단어가 서로에게 어떤 연관성을 가지고 있는지를 구하기 위해 사용.(문맥에 대한 정보를 잘 학습하기 위함) 

- 잔여학습(Residual Learning) : 성능향상을 위함. 어떤 값을 레이어를 거쳐서 반복적으로 단순하게 갱신하는 것이 아니라, 특정 레이어를 건너 뛰어서 복사가 된 값을 넣어주는 기법을 의미(Residual connection). 그러므로써 전체 네트워크는 기존 정보를 입력받으면서 추가적으로 잔여된 부분만 학습하도록 만들기 때문에 전반적인 학습 난이도가 낮아져 초기 수렴속도가 높아지고, 그로인해 글로벌옵티마를 찾을 확률이 높아진다.(아무튼 잔여학습을 사용하는게 성능향상에 좋다.)

![image](https://user-images.githubusercontent.com/74692845/137570616-30606007-52b6-411f-be45-09c8ecbd5565.png)

- 어텐션과 정규화(Normalization) 과정 반복하며 여러개의 레이어를 중첩해서 사용.(각각의 레이어는 서로 다른 파라미터를 가짐)

![image](https://user-images.githubusercontent.com/74692845/137570909-ed56ed10-f435-4b38-b6a3-b77d024949c7.png)


#### 2) Decoder

- 첫번째 Attention :  Self Attention. 인코더의 Attention과 마찬가지로 단어들이 서로가 서로에게 어떤 가중치를 가지는지 구하도록 만듦. 
- 두번째 Attention : Encoder-Decoder Attention. 인코더에 대한 정보를 attention함. 각각의 출력되고있는 단어가 소스문장에서의 어떤 단어와 연관성이 있는지 구해줌. (ex. teacher라는 단어가 전체문장 i'm a teacher.의 어떤 다른 단어와 연관성이 가장 높은지 알 수 있음)
- 트랜스포머에서는 마지막 인코더 레이어의 출력이 모든 디코더 레이어에 입력된다.
![image](https://user-images.githubusercontent.com/74692845/137571454-c938958c-74f3-4dc4-81ab-db1bda3debe7.png)
- <eos>가 나올 때까지 디코더를 이용.


##  Reference

- [https://www.youtube.com/watch?v=AA621UofTUA&t=5s] (유튜브 동빈나)
