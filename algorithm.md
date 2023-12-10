# 최적의 코딩을 결정하는 기본 알고리즘

## 1. 가장 기본이 되는 자료구조 : 스택과 큐

### 스택 자료구조
- 먼저 들어온 데이터가 나중에 나가는 형식(선입후출)의 자료구조
- 입구와 출구가 동일한 형채로 스택을 시각화할 수 있음
![Alt text](/reference_algorithm/image.png)

예시)
삽입(5) - 삽입(2) - 삽입(3) - 삽입(7) - 삭제() - 삽입(1) - 삽입(4) - 삭제()
![Alt text](/reference_algorithm/image-1.png)

```python
stack = []
stack.append(5)
stack.append(2)
stack.append(3)
stack.append(7)
stack.pop()
stack.append(1)
stack.append(4)
stack.pop()

print(stack[::-1]) # 최상단부터 [1, 3, 2, 5]
print(stack) # [5, 2, 3, 1]
```


### 큐 자료구조
- 먼저 들어온 데이터가 먼저 나가는 형식(선입선출)의 자료구조
- 큐는 입구와 출구가 모두 뚫려있는 터널과 같은 형태로 시각화할 수 있음
![Alt text](/reference_algorithm/image-2.png)

예시)
삽입(5) - 삽입(2) - 삽입(3) - 삽입(7) - 삭제() - 삽입(1) - 삽입(4) - 삭제()
![Alt text](/reference_algorithm/image-3.png)

```python
# 시간복잡도가 작음
from collections import deque

queue = deque()
queue.append(5)
queue.append(2)
queue.append(3)
queue.append(7)
queue.popleft()
queue.append(1)
queue.append(4)
queue.popleft()

print(queue) # 먼저 들어온 순서대로 출력 deque([3, 7, 1, 4])
queue.reverse() # 역순
print(queue) # 나중에 들어온 원소부터 출력 deque([4, 1, 7, 3])
```


## 2. 우선순위에 따라 데이터를 꺼내는 자료구조

### 우선순위 큐
- 우선순위 큐는 우선순위가 가장 높은 데이터를 가장 먼저 삭제하는 자료구조
- 우선순위 큐는 데이터를 우선순위에 따라 처리하고 싶을 때 사용
![Alt text](/reference_algorithm/image-4.png)


- 우선순위 큐를 구현하는 방법
    1. 단순히 리스트 이용
    2. 힙(heap)을 이용

- 데이터의 개수가 N개일 때, 구현방식에 따라 시간복잡도가 달라짐
![Alt text](/reference_algorithm/image-5.png)


#### 힙(Heap)의 특징
- 힙은 완전 이진 트리 자료구조
- 힙에서는 항상 루트 노트를 제거
- 최소 힙
    - 루트 노트가 가장 작은 값
    - 값이 작은 데이터가 우선적으로 제거
- 최대 힙
    - 루트 노드가 가장 큰 값
    - 값이 큰 데이터가 우선적으로 제거

![Alt text](/reference_algorithm/image-6.png)


#### 완전 이진 트리
- 루트 노드부터 시작하여 왼쪽 자식 노드, 오른쪽 자식 노드 순서대로 데이터가 차례대로 삽입되는 트리
![Alt text](/reference_algorithm/image-7.png)

#### 최소 힙 구성 함수(Min-Heapify)
- (상향식) 부모 노드로 거슬러 올라가며, 부모보다 자신의 값이 더 작은 경우에 위치를 교체
- 새로운 원소가 삽입되었을 때 O(logN)의 시간 복잡도로 힙성질 유지
- 원소가 제거되었을 때 O(logN)의 시간 복잡도로 힙성질 유지
![Alt text](/reference_algorithm/image-8.png)

예시)
```python
import sys
import heapq
input = sys.stdin.readline

def heapsort(iterable):
    h = []
    result = []
    for value in iterable
    heapq.heappush(h, value)

    for i in range(len(h)):
        result.append(heapq.heappop(h))
    return result

n = int(input())
arr = []

for i in range(n):
    arr.append(int(input()))

res = heapsort(arr)

for i in range(n):
    print(res[i])
```


## 3. 트리 자료 구조
- 트리 : 가계도와 같은 계층적인 구조럴 표현
    - 루트 노드 : 부모가 없는 최상위 노드
    - 단말 노드 : 자식이 없는 노드
    - 크기 : 트리에 포함된 모든 노드의 개수
    - 깊이 : 루트 노드부터의 거리
    - 높이 : 깊이 중 최댓값
    - 차수 : 각 노드(자식방향)의 간선 개수
- 트리의 크기가 N일 때, 전체 간선의 개수는 N-1개
![Alt text](/reference_algorithm/image-9.png)

### 이진 탐색 트리
- 이진 탐색이 동작할 수 있도록 고안된 효율적인 탐색이 가능한 자료구조
- 왼쪽 자식노드 < 부모노드 < 오른쪽 자식 노드
    - 부모 노드보다 왼쪽 자식 노드가 작다.
    - 부모 노드보다 오른쪽 자식 노드가 크다.

![Alt text](/reference_algorithm/image-10.png)

#### 트리의 순회
- 트리 자료구조에 포함된 노드를 특정한 방법으로 한 번씩 방문하는 방법
    - 트리의 정보를 시각적으로 확인가능

- 전위 순회 : 루트를 먼저 방문
- 중위 순회 : 왼쪽 자식을 방문한 뒤에 루트 방문
- 후위 순회 : 오른쪽 자식을 방문한 뒤에 루트 방문
![Alt text](/reference_algorithm/image-11.png)


## 4. 바이너리 인덱스 트리
- 바이너리 인덱스 트리(펜윅트리) : 2진법 인덱스 구조를 활용해 구간 합 문제를 효과적으로 해결할 수 있는 자료구조
ex)
7 = 00000000 0000111
-7 = 11111111 1111001 (7에서 1로 모두 전환 후 1을 더한 값)

- 0이 아닌 마지막 비트를 찾는 방법
: 특정한 숫자 K의 0이 아닌 마지막 비트를 찾기 위해서는 K & -K를 계산하면 됨

```python
n = 8

for i in range(n+1):
    print(i, '의 마지막 비트:', (i & -i))

# 0의 마지막 비트 : 0
# 1의 마지막 비트 : 1
# 2의 마지막 비트 : 2
# 3의 마지막 비트 : 1
# 4의 마지막 비트 : 4
# 5의 마지막 비트 : 1
# 6의 마지막 비트 : 2
# 7의 마지막 비트 : 1
# 8의 마지막 비트 : 8
```

- 트리구조 만들기 : 0이 아닌 마지막 비트 = 내가 저장하고 있는 값들의 개수
- 특정값을 변경할 때 : 0이 아닌 마지막비트만큼 더하면서 구간들의 값을 변경 (예시 3)
![Alt text](/reference_algorithm/image-12.png)

- 1부터 N까지의 합(누적 합) 구하기 : 0이 아닌 마지막 비트만큼 빼면서 구간들의 값의 합 계산
![Alt text](/reference_algorithm/image-13.png)

```python
import sys
input = sys.stdin.readline

# 데이터의 개수(n), 변경 횟수(m), 구간 합 계산 횟수(k)
n, m, k = map(int, input().split())

# 전체 데이터의 개수는 최대 1,000,000개
arr = [0] * (n+1)
tree = [0] * (n+1)

# i번째 수까지의 누적 합을 계산하는 함수
def prefix_sum(i):
    result = 0
    while i > 0:
        result += tree[i]
        # 0이 아닌 마지막 비트만큼 빼가면서 이동
        i -= (i&-i)
        return result

# i번째 수를 dif만큼 더하는 함수
def update(i, dif):
    while i <= n:
        tree[i] += dif
        i += (i&-i)
    return i

# start부터 end까지의 구간 합을 계산하는 함수
def interval_sum(start, end):
    return prefix_sum(end) - prefix_sum(start-1)

for i in range(1, n+1):
    x = int(input())
    arr[i] = x
    update(i, x)

for i in range(m+k):
    a, b, c = map(int, input().split())
    # 업데이트(update) 연산인 경우
    if a == 1:
        update(b, c-arr[b]) # 바뀐 크기(dif)만큼 적용
        arr[b] = c
    # 구간 합(interval sum) 연산인 경우
    else:
        print(interval_sum(b, c))
```


## 5. 선텍정렬과 삽입정렬
- 정렬(Sorting) : 데이터를 특정한 기준에 따라 순서대로 나열

#### 선택 정렬
- 처리되지 않은 데이터 중에서 가장 작은 데이터를 선택해 맨 앞에 있는 데이터와 바꾸는 것을 반복
```python
array = [7, 5, 9, 0, 3, 1, 6, 2, 4, 8]

for i in range(len(array)):
    min_index = i # 가장 작은 원소의 인덱스
    for j in range(i+1, len(array)):
        if array[min_index] > array[j] :
            min_index = j
    array[i], array[min_index] = array[min_index], array[i]

print(array) # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

#### 삽입 정렬
- 처리되지 않은 데이터를 하나씩 골라 적절한 위치에 삽입
```python
array = [7, 5, 9, 0, 3, 1, 6, 2, 4, 8]

for i in range(1, len(array)):
    for j in range(i, 0, -1): # 인덱스 i부터 1까지 1씩 감소하며 반복하는 문법
        if array[j] < array[j-1]:
            array[j], array[j-1] = array[j-1], array[j]
        else: # 자기보다 작은 데이터를 만나면 그 위치에서 멈춤
            break

print(array) # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] 
```


## 6. 퀵 정렬과 계수 정렬
#### 퀵정렬
- 기준 데이터를 설정하고 그 기준보다 큰 데이터와 작은 데이터의 위치를 바꾸는 방법
- 일반적으로 기준값(pivot)은 첫번째 데이터로 함
```python
array = [5, 7, 9, 0, 3, 1, 6, 2, 4, 8]

def quick_sort(array, start, end):
    if start >= end: # 원소과 1개인 경우 종료
        return
    pivot = start
    left = start + 1
    right = end
    while(left <= right):
        # 피벗보다 큰 데이터를 찾을 때까지 반복
        while (left <= end and array[left] <= array[pivot]):
            left += 1
        # 피벗보다 작은 데이터를 찾을 때까지 반복
        while (right > start and array[right] >= array[pivot]):
            right -= 1
        if(left > right): # 엇갈렸다면 작은 데이터와 피벗을 교체
            array[right], array[pivot] = array[pivot], array[right]
        else: # 엇갈리지 않았다면 작은 데이터와 큰 데이터를 교체
            array[left], array[right] = array[right], array[left]
    # 분할 이후 왼쪽 부분과 오른쪽 부분에서 각각 정렬 수행
    quick_sort(array, start, right-1)
    quick_sort(array, right+1, end)
quick_sort(array, 0, len(array)-1)

print(array) # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

```python
array = [5, 7, 9, 0, 3, 1, 6, 2, 4, 8]

def quick_sort(array):
    if len(array) <= 1:
        return array
    pivot = array[0] # 피벗은 첫번재 원소
    tail = array[1:]

    left_side = [x for x in tail if x <= pivot] # 분할된 왼쪽 부분
    right_side = [x for x in tail if x > pivot] # 분할된 오른쪽 부분

    # 분할 이후 왼쪽 부분과 오른쪽 부분에서 각각 정렬 수행하고 전체 리스트 반환
    return quick_sort(left_side) + [pivot] + quick_sort(right_side)

print(quick_sort(array)) # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

#### 계수정렬
- 동일한 값을 가지는 데이터가 여러 개 등장할 때 효과적으로 사용
- 데이터가 0, 999,999인 2개의 값만 있다고 했을 때 매우 비효율적

```python
array = [7, 5, 9, 0, 3, 1, 6, 2, 9, 1, 4, 8, 0, 5, 2]
count = [0] * (max(array) + 1)

for i in range(len(array)):
    count[array[i]] += 1

for i in range(len(count)):
    for j in range(count[i]):
        print(i, end=' ')
```


## 7. DFS와 BFS
#### DFS
- 깊이우선탐색(DFS) : 깊은 부분을 먼저 탐색
1. 탐색 시작 노드를 스택에 삽입하고 방문 처리
2. 스택의 최상단 노드에 방문하지 않은 인접한 노드가 하나라도 있으면 그 노드를 스택에 넣고 방문처리\
    방문하지않은 인접노드가 없으면 스택에서 최상단 노드를 꺼냄
3. 더 이상 2번의 과정을 수행할 수 없을 때까지 반복

![Alt text](/reference_algorithm/image-15.png)

```python
dfs dfs(graphm v, visited):
    visited[v] = True
    print(v, end=' ')
    for i in graph[v]:
        if not visited[i]:
            dfs(graph, i, visited)

graph = [
    [],
    [2, 3, 8],
    [1, 7], 
    [1, 4, 5],
    [3, 5],
    [3, 4],
    [7],
    [2, 6, 8],
    [1, 7],
]

visited = [False] * 9
dfs(graph, 1, visited)
# 1 2 7 6 8 3 4 5
```


#### BFS
- 너비우선탐색(BFS) : 가까운 노드부터 우선적으로 탐색
- 큐 자료구조를 이용
1. 탐색 시작 노드를 큐에 삽입하고 방문 처리
2. 큐에서 노드를 꺼낸 뒤에 해당 노드의 인접 노드 중에서 방문하지 않은 노드를 모두 큐에 삽입하고 방문 처리
3. 더 이상 2번의 과정을 수행할 수 없을 때까지 반복

![Alt text](/reference_algorithm/image-15.png)

```python
from collections import deque

def bfs(graph, start, visited):
    queue = deque([start])
    visited[start] = True
    while queue:
        v = queue.popleft()
        print(v, end=' ')
        for i in graph[v]:
            if not visited[i]:
                queue.append(i)
                visited[i] = True

graph = [
    [],
    [2, 3, 8],
    [1, 7], 
    [1, 4, 5],
    [3, 5],
    [3, 4],
    [7],
    [2, 6, 8],
    [1, 7],
]

visited = [False] * 9
bfs(graph, 1, visited)
# 1 2 3 8 7 4 5 6
```


## 8. 다익스트라 알고리즘
1. 출발노드 설정
2. 최단 거리 테이블 초기화
3. 방문하지 않은 노드 중에서 거리가 가장 짧은 노드를 선택
4. 해당 노드를 거쳐 다른 노드로 가는 비용을 계산하여 최단거리테이블 갱신
5. 위 과정에서 3번과 4번을 반복

