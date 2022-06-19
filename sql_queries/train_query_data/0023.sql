\o out.txt
SET random_page_cost = 32;

\timing

SELECT p_partkey,
	ps_suppkey
FROM part, partsupp
WHERE p_name LIKE '%blush%';

\timing
