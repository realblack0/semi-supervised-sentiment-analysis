# 감성사전 기반 준지도학습 감성분석

## 모델 설명

**Pointwise Mutual Information**

두 어휘가 동시에 나타날 확률을 두 어휘가 각각 독립적으로 나타날 확률로 나누고 로그를 씌워 두 어휘 간의 상호 정보량을 측정한다.

$PMI(x;y)=log\frac{P_{X,Y}(x, y)}{P_{X}(x)P_{Y}(y)}$

**Semantic Orientation**

어휘의 긍정 상호정보량에서 부정 상호정보량을 차감하여 어휘의 감성을 측정한다.

$SO(x)=PMI(x;positive) - PMI(x;negative)$

**Semi-Supervised**

모든 문서에 라벨이 없는 경우에도 학습할 수 있는 준지도학습 방식을 사용하였다. 문서의 라벨($positive$, $negative$) 대신, 극성 토큰($Seed_{pos}$, $Seed_{neg}$)과 각 토큰의 상호정보량(PMI)을 측정함으로써 감성 사전을 구축한다. 극성 토큰은 하이퍼파라미터에 해당한다. ($n$은 긍정 seed 토큰의 개수, $m$은 부정 seed 토큰의 개수)

$SO(x) = \sum_{pos=pos_1}^{pos_n} {PMI(x; Seed_{pos})} - \sum_{neg=neg_1}^{neg_m} {PMI(x; Seed_{neg})}$

## 간단 튜토리얼

### 학습 데이터

데이터는 pandas의 DataFrame을 이용하며, 학습을 위해서는 다음과 같은 column이 반드시 포함되어야 한다.

- content : 학습 문서; 문자열 데이터
- Trend : 데이터의 라벨; 긍정 라벨은 P, 부정 라벨은 N, 그 외 상관 없음
- 데이터 준비 예시:

|index|content|Trend|
|---|---|---|
|0|Korea’s coronavirus infections drop back to two-digit figure, total now at 8,652|P|
|1|There are 1346 hospitalized patients with symptoms, 295 are in intensive care and 1065 are in home isolation.|N|
|2|The novel Covid-19 endemic, which was declared a public health emergency of international concern by WHO on January 30, has spread from the Wuhan Province in China to 25 countries. So far, there are 76,366 cases globally with more than 2,300 deaths recorded.||
|3|NAN, reports that as the COVID1-19 pandemic grows, health officials continue to monitor the number of confirmed cases. Globally there are 468,156 cases and 21,180 deaths from COVID-19 outbreaks.|N|
|...|...|...|

### 모듈 소개

- model : PMI 모델
- process : 전처리, 토크나이저 관련 모듈
- sequential : 모델의 필수 인자 모듈; 모델 내에서 process를 순차적으로 실행시키는 파이프라인 역할
- utils : 데이터 로드 관련 함수

### 모델 구성

모델은 `Normalize`, `Tokeninze`, `Cleansing`, `Postag` 인스턴스를 필수 인자로 한다.

```python
pmi = PMI(normalize, tokenize, cleansing, postag)
```

### 모델 주요 기능

- process : 전처리 및 토큰화 후 TDM(Term Document Matrix)을 구축한다.

- post_process : process 이후 토큰을 분석하여 불용어를 추가한다. 예를 들어, 빈도 상하위 20% 토큰을 제외할 수 있다.

- info : total, pos, neg 별로 토큰의 총 개수와 unique한 토큰 개수를 출력한다.

```python
pmi.info()
```

>total  
>
>total words: 7649153  
>unique words: 61619  
>
>positive  
>
>total words: 4997  
>unique words: 1693  
>
>negative  
>
>total words: 367712  
>unique words: 14997

- most_common : total, pos, neg 별 빈도 상위 n개의 토큰을 순서대로 출려해 극성 단어 선별을 돕는다.

```python
pmi.most_common(100)
```

>상위 100개 단어
순위      total     pos     neg   
\----------------------------------------  
0번째: ('covid', 212780)    ('cases', 121)    ('covid', 7555)  
1번째: ('china', 104872)    ('covid', 104)    ('cases', 5879)  
2번째: ('said', 94906)    ('new', 77)    ('said', 5750)  
3번째: ('health', 87720)    ('china', 77)    ('china', 5670)  
4번째: ('cases', 87615)    ('said', 76)    ('health', 5324)  
5번째: ('people', 63955)    ('confirmed', 73)    ('wuhan', 4110)  
6번째: ('virus', 61348)    ('health', 56)    ('people', 3973)  
7번째: ('new', 57449)    ('hubei', 48)    ('virus', 3942)  
8번째: ('confirmed', 42446)    ('province', 48)    ('confirmed', 3926)  
9번째: ('wuhan', 41268)    ('virus', 41)    ('new', 3813)  
...

- plot : total, pos, neg 별 wordcloud를 그려 극성 단어를 선별을 돕는다.

```python
pmi.plot()
```
![total wordcloud](https://user-images.githubusercontent.com/50395556/81842003-4391b400-9586-11ea-9ec8-0cd6b5427649.png)

![pos wordcloud](https://user-images.githubusercontent.com/50395556/81842112-720f8f00-9586-11ea-9dbb-7664baa28d28.png)

![neg wordcloud](https://user-images.githubusercontent.com/50395556/81842067-5c01ce80-9586-11ea-8681-caaa5a96687b.png)

- make_seeds : 단어사전 내에서 입력받은 극성 단어와 일치하는 토큰 또는 파생형 토큰을 찾는다.

- fit : 극성 단어를 기반으로 감성 사전을 구축한다.

- predict : 감성 사전을 기반으로 학습 데이터의 감성을 예측한다.

## 의존 패키지

>- pandas 1.03
>- matplotlib
>- nltk 3.4.5
>- tqdm 4.36.1