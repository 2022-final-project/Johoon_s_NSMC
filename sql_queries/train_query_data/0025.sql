\o out10.txt
SET random_page_cost = 1;


\timing

select
	c_custkey,
	c_name,
	sum(l_extendedprice * (1 - l_discount)) as revenue
from
	customer,
	orders,
	lineitem
where
	c_custkey = o_custkey
	and l_orderkey = o_orderkey
	and l_returnflag = 'R'
group by
	c_custkey,
	c_name,
	c_acctbal,
	c_phone
order by
	revenue desc;

\timing

-- Default : LIMIT 20 
