\o out10.txt
SET random_page_cost = 16;

\timing

select
	c_custkey,
	c_name,
	n_name,
	sum(l_extendedprice * (1 - l_discount)) as revenue
from
	customer,
	orders,
	lineitem,
	nation
where
	c_custkey = o_custkey
	and l_orderkey = o_orderkey
	and l_returnflag = 'R'
	and l_shipdate < date '1995-03-01'
group by
	c_custkey,
	c_name,
	c_acctbal,
	c_phone,
	n_name
order by
	revenue asc;

\timing

-- Default : LIMIT 20 
