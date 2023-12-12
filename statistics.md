## 1. 확률의 개념과 특징
- 통계학 : 데이터에 담겨진 표면적인 정보를 정확히 요약하고 그 내면에 담긴 의미를 추록하고 해석하기 위한 도구적인 성격의 학문
- 모수 : 알고자 하는 미지의 정보 (어느 후보의 지지율, 불량률 등) -> 데이터를 활용해 예측

#### 확률의 기본 개념
- 확률 : 어떤 사건이 발생할 가능성을 0에서 1사이의 숫자로 표현
- 확률 모형 : 시행을 반복할 때마다 나오는 결과가 우연에 의존하여 매번 달라지는 현상 또는 실험(확률 실험)에 대한 수리적 모형
- 표본공간 : 확률 실험에서의 모든 관찰 가능한 결과의 집합(S로 표기)\
예) 동전 (앞, 뒤) 주사위 (1, 2, 3, 4, 5, 6)
- 사건 : 표본공간의 임의의 부분집합 (A, B등으로 표기)

#### 확률의 정의 및 성질
- 고전적 접근 : n개의 실험결과로 구성된 표본공간에서 각 실험결과가 일어날 가능성이 같은 경우 m개의 실험결과로 구성된 사건 A의 확률\
P(A) = m / n
- 상대적 비율에 의한 접근
    - n번의 반복된 실험 중 어떤 사건 A가 발생한 횟수를 m이라고 할 때 사건 A의 상대빈도는 m / n 으로 구해짐
    - 이 실험의 반복횟수 n을 무한히 증가했을 때, 사건 A의 상대빈도가 수렴하는 값을 사건 A의 확률로 정의

- 확률의 공리
    1. 임의의 사건 A에 대하여 P(A) >= 0
    2. P(S) = 1
    3. 표본공간 S에 정의된 서로 상호배반인 사건 A1, A2 ... 에 대하여 P(A1 U A2 ...) 합집합 = P(A1) + P(A2) + ... 가 성립
![Alt text](/reference_statistics/image-1.png)

- 공리적 접근방식 : 표본공간을 정의역으로 하며, 3가지 공리를 만족하는 함수를 확률로 정의
- 여사건의 확률
    - P[A^c] 여집합 : 사건 A를 제외한 나머지 사건의 확률
    - P[A^c] = 1 - P[A]
    ![Alt text](/reference_statistics/image-2.png)
- 곱사건의 확률
    - P[A 교집합 B] : 사건 A와 사건 B가 동시에 발생할 확률
    ![Alt text](/reference_statistics/image-3.png)
- 합사건의 확률
    - P[A U B] 합집합 : 사건 A 또는 사건 B가 발생할 확률
    - P[A U B] 합집합 = P[A] + P[B] - P[A 교집합 B]
- 조건부 확률\
: A와 B가 표본공간 S상에 정의되어 있으며 P(B) > 0라고 가정\
이 때 사건 B가 일어났다는 가정하의 사건 A가 일어날 조건부 확률
    - P(A | B) = P(A 교집합 B) / P(B)

    예) A = 주사위에서 3 이하의 숫자(1, 2, 3)\
    B = 주사위에서 짝수(2, 4, 6)\
    결과가 짝수일 때 이 결과가 2일 확률\
    P(A | B) = (1 / 6) / (1 / 2) = 1 / 3
    ![Alt text](/reference_statistics/image-4.png)
- 독립사건 : 두 사건 A와 B가 다음 중 하나를 만족시키면 서로 독립
(단, P(A) > 0, P(B) > 0)
    1. P(A | B) = P(A)
    2. P(A 교집합 B) = P(A) * P(B)
    3. P(B | A) = P(B)

## 2. 베이즈 정리
#### 표본공간의 분할과 전확률 공식
- 표본공간의 분할 : B1, B2, ..., Bk(k개의 원인)가 다음 조건을 만족하면 표본 공간 S의 분할
    - 서로 다른 i, j에 대해 Bi 교집합 Bj = 공집합(상호배반)
    - B1 U B2 U ... U Bk (합집합) = S
    ![Alt text](/reference_statistics/image-5.png)
- 전확률공식
    - 사건 B1, B2, ..., B는 상호배반이며 B1 U B2 U ... U Bk (합집합) = S 일 때, S에서 정의되는 임의의 사건 A에 대해 다음이 성립
    - P(A) = P(A 교집합 B1) + ... + P(A 교집합 Bk)\
    = P(B1)P(A | B1) + ... + P(Bk)P(A | Bk)
    ![Alt text](/reference_statistics/image-6.png)

#### 베이즈 정리
- 베이즈정리 : 데이터라는 조건이 주어졌을 때의 조건부 확률을 구하는 공식
- 사건 B1, B2, ..., Bk는 상호배반이며, B1 U B2 U ... U Bk (합집합) = S라고 함
- 이 때 사건 A가 일어났다는 조건 하에 사건 Bi가 일어날 확률\
P(Bi | A) = P(A 교집합 Bi) / P(A) = P(Bi)P(A | Bi) / P(B1)P(A | B1) + ... + P(Bk)P(A | Bk)
![Alt text](/reference_statistics/image-7.png)
![Alt text](/reference_statistics/image-8.png)


## 3. 확률변수와 확률분포, 분포의 특성치
#### 확률변수
- 확률변수 : 표본공간에서 정의된 실수값 함수
    - 이산형 확률변수 : 확률변수가 취할수 있는 값이 셀 수 있는 경우(예. 고객 사고건 수, 불량 수)
    - 연속형 확률변수 : 주어진 구간에서 모든 실수 값을 취할 수 있어 셀수 없는 경우(예. 시간, 길이)

#### 확률분포
- 확률질량함수(이산형 확률변수)\
: 확률변수 X가 이산형인 경우 X가 취할 수 있는 값 x1, x2, ..., xn의 각각에 대하여 확률 P[X = x1], P[X = x2], ..., P[X = xn]을 대응시켜주는 관계\
f(x)는 X 내에서 x가 일어날 확률\
f(xi) = P[X = xi], i = 1, 2, ..., n
    1. 모든 i = 1, 2, ..., n에 대해 0 <= f(xi) <= 1
    2. 모든 f(xi)의 합 = 1

- 확률밀도함수(연속형 확률변수)\
: 확률변수 X가 연속형인 경우 X가 가질 수 있는 구간(-무한, 무한)위에서의 함수 f(x)가 다음을 만족할 때
![Alt text](/reference_statistics/image-9.png)
![Alt text](/reference_statistics/image-10.png)

- 누적분포함수\
: -X의 확률밀도함수가 f(x)일 때 X의 누적분포함수 F(x)는 X <= x인 모든 X에 대한 f(x)의 적분값이 됨
![Alt text](/reference_statistics/image-11.png)
    1. F(-무한대) = 0, F(무한대) = 1
    2. x가 증가할 때 F(x)도 증가하며, F(x)는 음의 값을 가질 수 없음

#### 확률분포의 특성치
- 기대값 : 분포의 무게중심, 중심 위치를 나타냄
![Alt text](/reference_statistics/image-12.png)
- 분산 : 분포의 산포
![Alt text](/reference_statistics/image-13.png)
- 표준편차 : 분산의 제곱근(단위가 보정됨)
![Alt text](/reference_statistics/image-14.png)


## 4. 이항분포, 포아송분포, 지수분포, 감마분포
#### 이항분포
- 베르누이시행 : 아래 두 가지 조건을 만족하는 실험
    - 매 시행마다 '성공' 또는 '실패'의 두 가지 결과만 가짐
    - '성공'의 확률이 p로 일정\
예) 동전 실험
- 이항 확률변수가 고려되는 실험 : 베르누이시행을 독립적으로 n번 반복하는 실험
    - X : n번 시행 중 '성공'의 횟수로 정의
    - 이항확률 변수 x = 0, 1, ..., n의 값을 가짐
    - ![Alt text](/reference_statistics/image-15.png)
    - 이 경우 X~Bin[n, p]
![Alt text](/reference_statistics/image-16.png)

- 이항분포의 특성치\
: X~Bin[n, p]일 때,
    - 기대값(평균값) E[X] = np
    - 분산 V[X] = np(1-p)

#### 포아송 분포
- 단위시간(t=1), 포아송확률과정을 따르는 사건 A가 발생하는 횟수 X일 때\
f(x) = P(X = x) = exp(-m) * m^x / x! (x = 0, 1, 2, ...)
- 이 경우 X~POI[m]

- 포아송 분포의 특성치
    - X~POI[m]인 경우 : E[X] = V[X] = m

    
예) 하루 평균 고속도로 교통사고 발생 2건 일 때 하루 3건일 확률\
X~POI[2]이고 P[X = 3] = 위의 식 활용

#### 지수분포
- 단위구간에서 평균발생횟수가 m인 포아송 확률과정을 따르는 사건 A가 한번 일어난 뒤 그 다음 또 일어날 때까지 걸리는 시간 W로 정의
![Alt text](/reference_statistics/image-17.png)
X~EXP[람다]
- 지수분포의 특성치
    - 포아송모수와 지수모수는 역의 관계 : 단위구간 내 평균발생횟수가 m인 포아송과정을 따르는 사건은 사건 사이 소요시간의 평균이 람다 = 1/m

#### 감마분포
- 감마확률변수 X의 확률밀도함수는 양수인 세타와 k에 대해
![Alt text](/reference_statistics/image-18.png)
이 경우 X~GAMMA[k, 세타]
    - 감마분포의 특성치\
        - X~GAMMA[k, 세타]인 경우\
        E[X] = k*세타\
        V[X] = k * 세타^2


## 5. 정규분포, 표준정규분포
#### 정규분포
- 확률변수 X가 평균이 u, 분산이 시그마^2 이고, 다음의 확률밀도함수를 가질 때 X는 정규분포를 따름
![Alt text](/reference_statistics/image-19.png)
이 경우 X~N [u, 시그마^2]
![Alt text](/reference_statistics/image-20.png)

    - u는 분포의 중심
    - u를 중심으로 대칭이고 u에서 가장 큰 값이 되는 하나의 봉우리만 가짐
    - 시그마^2이 크면 분포의 산포가 커짐

- 정규분포의 특성치\
X~N[u, 시그마^2]인 경우
    - E[X] = u
    - V[X] = 시그마^2

#### 표준정규분포
- X~N[u, 시그마^2]일 때, 정규분포의 선형불변성에 의해 Z = (X-u) / 시그마 ~N[0, 1]이 되며, 이 때 평균이 0, 분산이 1인 정규분포
![Alt text](/reference_statistics/image-21.png)

- 표준정규 확률변수의 (1 - a)분위수 : Z_a\
Z~N[0, 1]일 때 P[Z < c] = 1 - a 를 만족하는 Z의 (1-a)분위수 c를 Z_a로 표기
![Alt text](/reference_statistics/image-22.png)


## 6. 카이제곱분포, t분포, f분포
#### 카이제곱분포 : 표준정규들의 제곱합이 가지는 분포
- Z1, Z2, ..., Zk가 k개의 서로 독립인 표준정규 확률변수(Zi~N[0, 1], i=1, 2, ..., k)라고 할 때,\
X = Z1^2 + Z2^2 + ... + Zk^2가 따르는 분포를 자유도가 k인 카이제곱분포라고 정의
![Alt text](/reference_statistics/image-23.png)
이 경우 X~x^2[k]

- 카이제곱분포의 특성치:\
X~x^2[k]인 경우
    - E[X] = k
    - V[X] = 2k
    ![Alt text](/reference_statistics/image-24.png)

- 카이제곱 확률변수의 (1-a)의 분위수 : x^2_a, k\
X~x^2[k]일 때, P[X > c] = a를 만족하는 X의 (1-a)분위수 c
![Alt text](/reference_statistics/image-25.png)


#### t분포 : 표준정규에 평균이 가지는 분포
- Z가 표준정규확률변수 Z~N[0, 1]일 때, U가 자유도가 k인 카이제곱확률변수 U~x^2[k]이며, Z와 X는 서로 독립이라고 할 때 X = Z / 루트(U/k) 가 따르는 분포를 자유도가 k인 t분포라고 정의
![Alt text](/reference_statistics/image-26.png)
![Alt text](/reference_statistics/image-27.png)

- t분포의 특성치\
X~t[k]인 경우
    - E[X] = 0
    - V[X] = k / k-2 (단 k > 2)

- t분포 확률밀도함수 개형
X~t[k]인 경우
    - 가운데 0을 중심으로 대칭인 종모양의 분포
    - 표준정규분포보다 꼬리가 두꺼움
    - 자유도 k가 커짐에 따라 산포가 줄어들어 표준정규분포로 수렴

- t확률변수의 (1-a)분위수 : t_a,k
X~t[k]일 때, P[X > c] = a를 만족하는 X의 (1-a)분위수 c
![Alt text](/reference_statistics/image-28.png)

#### f분포 : 2개의 카이제곱분포의 비율이 만들어내는 분포
- U가 자유도가 k1인 카이제곱 확률변수 U~x^2[k1]이며 V가 자유도가 k2인 카이제곱 확률변수 V~x^2[k2]이고 U와 V는 서로 독립이라고 할 때 X = (U/k1) / (V/k2)가 따르는 분포를 자유도가 k1, k2인 F분포
![Alt text](/reference_statistics/image-29.png)
X~F[k1, k2]라고 함

- F분포의 특성치\
X~F[k1, k2]인 경우
![Alt text](/reference_statistics/image-30.png)
![Alt text](/reference_statistics/image-31.png)

- F 확률변수의 (1-a)분위수 : F_a,k1,k2\
X~F[k1, k2]일 때, P[X > c] = a를 만족하는 X의 (1-a)분위수 c
![Alt text](/reference_statistics/image-32.png)

