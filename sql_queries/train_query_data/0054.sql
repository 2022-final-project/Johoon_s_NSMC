\o out.txt
SET random_page_cost = 1;

\timing

SELECT p_partkey,
	avg(p_size),
	count(*)
FROM part, 
	partsupp
WHERE p_name LIKE '%blush%'
	AND p_brand IN ('Brand#32', 'Brand#42', 'Brand#13')
	and p_size < 30
GROUP BY p_partkey;

\timing
