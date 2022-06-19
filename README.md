1. query_data_refining.py 
    - `sql_to_text()`
        - train data로 사용되는 sql파일들을 하나하나 불러오면서 한줄화 한다.
    - `query_refining1()`
        - column, table data preprocessing을 위한 text file을 통해
          column, table을 token화한 dictionary를 만든다.
        - alias로 사용된 word들을 만약 column을 뜻하는 alias인 경우
          alias dictionary에 추가한다.
    - `query_refining2()`
        - 그 후 괄호, 점, where절에서 사용되는 data value들을 지운다.
    - `table_column_preprocessing()`
        - query_refining1() method에서 제작된 column, table dictionary를 통해
          각 query들의 column, table name을 token화 한다.
    - `make_data_with_label()`
        - refining된 query data와 label text data를 통해 train_data.txt를 제작한다.
    - `make_vocab()`
        - refining된 query data text를 통해 그 text file에 있는 word를 모두 vocab.txt에 넣는다.
          이 때, 가장 많이 사용된 순서대로 넣는다.