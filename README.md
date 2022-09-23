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

3. <U>**Query data preprocessing, Tokenizer 구현**</U>
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
     - 정제시 사용할 단어들을 미리 list로 구현

4. 