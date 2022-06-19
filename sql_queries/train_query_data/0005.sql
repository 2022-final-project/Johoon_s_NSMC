\o out5.txt
SET random_page_cost = 32;

\timing

select
	n_name,
	sum(l_extendedprice * (1 - l_discount)) as revenue
from
	customer,
	orders,
	lineitem,
	supplier,
	nation,
	region
where
	c_custkey = o_custkey
	and l_orderkey = o_orderkey
	and l_suppkey = s_suppkey
	and c_nationkey = s_nationkey
	and s_nationkey = n_nationkey
	and n_regionkey = r_regionkey
	and r_name = 'AISA'
	and o_orderdate >= date '1995-03-01'
	and o_orderdate < date '1995-03-01' + interval '1 year'
group by
	n_name
order by
	revenue desc
LIMIT 1;

\timing
