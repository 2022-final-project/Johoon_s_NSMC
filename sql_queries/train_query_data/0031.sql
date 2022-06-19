\o out3.txt
SET random_page_cost = 32;
\timing

select
	l_orderkey,
	sum(l_extendedprice * (1 - l_discount)) as revenue,
	count(o_orderdate)
from
	orders,
	lineitem
where
	l_orderkey = o_orderkey
	and o_orderdate < date '1995-03-01'
	and l_shipdate > date '1995-03-01'
group by
	l_orderkey,
	o_orderdate
order by
	revenue desc,
	o_orderdate;

\timing
