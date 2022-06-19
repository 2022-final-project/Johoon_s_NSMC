\o out.txt
SET random_page_cost = 32;

\timing

SELECT p_partkey,
	avg(p_size)
FROM part, partsupp
WHERE p_name LIKE '%blush%'
	AND p_brand IN ('Brand#32', 'Brand#42', 'Brand#13')
GROUP BY p_partkey;

\timing
