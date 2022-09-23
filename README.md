# [S/W 개발 환경]
* OS : Windows 11
* IDE : Visual Studio Code
* Used tools : PyTorch, PostgreSQL, TPC-H, Django
* Used languages : Python, PostgreSQL, HTML, CSS
<br></br>

# [설명]
1. <U>**TPC-H Benchmark를 이용하여 Database에 대한 Metadata 확보**</U>
   - 확보된 Metadata
     - `8개의 Tables`
     - `Table 들에 속해있는 Dataset (사용된 Dataset 용량 : 0.01GB ~ 10GB)`
     - `22개의 Queries`

2. <U>**22개의 Query 이외에 학습을 위한 추가적인 Query 제작**</U>

3. train_labels.txt 구현
   - PostgreSQL 에서 random_page_cost의 값을

     |1.0|2.0|4.0|8.0|16.0|32.0|
     |---|---|---|---|---|---|

     이와 같이 6개의 값을 측정하여 빠른 실행시간을 내는 cost value 를 탐색
   - 빠른 실행시간을 내는 경우에는 `1`, 아닌 경우에는 `0`을 부여한다.
   - 만약 1.0, 8.0의 값을 가질때 빠른 실행시간을 내는 경우 label을 `100100`으로 선정

4. <U>**Query data preprocessing, vocab.txt 구현**</U>
   - `train_query_data_refining.py`
   - 학습을 시키기 위해서 Query data의 정제가 필요
   - Ex) 
      ```
      [정제 전]

      SELECT name, avg(age)
      FROM users AS u
      WHERE name LIKE 'Choi%'
            AND salary < 500
      ORDER BY name ASC
      ```

      ```
      [정제 후]

      SELECT c1 avg c2
      FROM t1
      WHERE c1 LIKE
            AND c3
      ORDER BY c1 ASC
      ```
     - `train_data.txt` 구현
     - 정제시 사용할 단어들을 미리 list로 구현
     - vocab.txt를 생성하는 source 구현

5. <U>**`modeling.py`**</U>
   - train_label.txt 파일을 참고하여 labels list를 제작한다.
   - label은 6차원 vector로 재정의 한다.

      |`100100`|&rarr;|[1, 0, 0, 1, 0, 0]|
      |---|---|---|

