---
title: Deep Learning Specialization
date: 2021-02-07 18:02:57
category: coursera
draft: false

---

앤드류 응 선생님의 딥러닝 전문가 코세라 강의. 구글 부트캠프 수료 과정에서 중점적으로 들어야하는 강의이다. 2017년에 나온 강의기에 최신 트렌드를 담았다고 볼 수는 없으나 딥러닝 기초를 확립하기에 좋은 강의이다. 이미 머신러닝과 딥러닝 기초적인 내용은 개인적으로 충분하다고 오만하게 판단했지만, 첫주차 강의를 듣고 그 생각이 바뀌었다. 다른 사람들도 이 강의를 통해 확실히 딥러닝 AI 지식 기초를 다질 수 있을 것이라 확신한다.

코스는 총 5개로 구성되어 있으며, 수료하면서 요약했던 내용을 개인적인 복습 차원에서 요약 및 정리를 하려고 한다. 역전파 등 수식 과정은 이 강의 내용에서나 유튜브에서나 훨씬 더 잘 설명되어 있다고 판단되어, 이론 위주로 내용을 정리해보았다. 물론 필연적으로 수식이 등장하겠지만 깔끔하게 보여지기 위해 강의 노트를 옮겨왔다. 기본적인 annotation은 안다고 가정하였다.

## Course 1. Neural Networks and Deep Learning

## Logistic Regression
딥러닝을 알기 위해서는 먼저 `Logistic Regression` 을 알아야 한다. `Logistic Regression` 은 어떤 x (features) Y를 예측하는데 사용되는 기본 선형 확률 모델이다. 강의에 나왔던 예시로 예를 들자면 **어떤 사진**이 주어졌을 때 **사진이 고양이인지, 강아지인지 판단하는 작업**을 의미한다. 고양이인지, 강아지인지에 대한 분류 정확도를 로지스틱 회귀 모델은 0에서 1 사이의 확률로 표현해준다. 

이를 수식으로 간단히 정리하자면 예측값인 $\hat{y}$은 다음과 같다.
$$\hat{y} = \sigma(W^Tx + b)$$

여기서 중요한 부분은 `Sigmoid`함수 부분인데, 아래 그림은 `Sigmoid`함수를 나타낸다. 어떠한 값을 받아도 0 ~ 1 사이의 값으로 변환해주는 것을 볼 수 있다.

![logistic regression 이미지 검색결과](https://miro.medium.com/max/2400/1*RqXFpiNGwdiKBWyLJc_E7g.png)


**즉, 로지스틱 회귀 모델은 0 ~ 1 사이의 확률값으로 어떤 예측에 대한 수치를 표현하는 것이 핵심이다.** 만약 받은 값 z가 매우 클 경우 그 값은 1에 가까울 것이고 매우 작을 경우 0에 가까워질 것이다.

---

여기서 로지스틱 회귀 모델로부터 나온 예측값을 $y$가 아닌 $\hat{y}$으로 표현했다. 이유는 이 수치는 어디까지나 **정답**이 아닌 **예측값**이기 때문이다. 분류 모델의 정확도를 평가 하기 위해 로지스틱 회귀 모델에서는 정답과 예측값 간의 차이를 산출한 뒤, 이를 최소화하는 것을 목적으로 한다.  회귀분석 모델에서 파라미터 W와 b를 훈련시키기 위해서는 먼저 **Loss function**를 정의해야 한다. 단순히 아래처럼 예측값과 정답 간 차이를 Loss function으로 설정할 수도 있다. 
$$L(\hat{y}, y) = \frac{1}{2}(\hat{y} - y)^2$$
그러나 이렇게 하지 않는 이유는, `gradient descent (모델이 훈련과정에서 찾아야할 미분값)의 최적값을 찾기 힘들다는 단점 때문에 비슷한 역할을 하는 로그 함수를 사용한다. 

$$L(\hat{y}, y) = -(ylog\hat{y} + (1-y)log(1-\hat{y}))$$

직관적으로 위 식은 만약 y가 1일 경우 Loss function의 값이 최소가 되기 위해서는 $\hat{y}$를 커지는 방향으로 학습해야 하고, y가 0일 경우 $\hat{y}$이 작아지는 방향으로 학습이 진행되기 때문에 적절하다. 

유사한 개념으로 Cost function이 등장하는데, 위에서 계속 언급했던 Loss function과 개념적으로는 차이가 거의 없다. 다만 차이가 있다면 Loss function은 단일(single) training example에 대한 error 이며, Cost function은 전체 training set에 대한 error의 평균으로 설명하였다. 

---
Gradient Descent


여기서 두가지 개념이 나오는데, Loss function과 cost function 두 개념이 나온다. 
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTI2NDIwNTk4OSwtNDc5NTU1NzA0LC0yMD
I1MDM3MjA4LDExMDY3MjI1MTksOTA4NzU4MDQyLC0xMzczNTgw
MjVdfQ==
-->