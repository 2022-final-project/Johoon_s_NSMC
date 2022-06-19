\o out.txt
SET random_page_cost = 1;

\timing

select
	avg(l_extendedprice)
from
	lineitem,
	part,
	partsupp
where
	l_shipdate < date '1995-03-01'
	and p_partkey = ps_partkey
	and p_partkey = l_partkey
	and p_brand LIKE '%2%'
	and p_container = 'LG PACK'
	and l_quantity < (
		select
			0.2 * avg(l_quantity)
		from
			lineitem
		where
			l_partkey = p_partkey
	);

\timing
