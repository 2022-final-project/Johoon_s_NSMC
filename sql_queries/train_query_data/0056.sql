\o out.txt
SET random_page_cost = 32;
\timing

select
	ps_partkey,
	sum(ps_supplycost * ps_availqty) as value,
	n_name
from
	partsupp,
	supplier,
	nation,
	lineitem
where
	ps_suppkey = s_suppkey
	and s_nationkey = n_nationkey
	and n_name like 'UNITED%'
	and s_address in ('%0%', '%A%', '%a%')
	and l_shipdate < date '1995-03-01'
group by
	n_name,
	ps_partkey having
		sum(ps_supplycost * ps_availqty) > (
			select
				sum(ps_supplycost * ps_availqty) * 0.00004
			from
				partsupp,
				supplier,
				nation
			where
				ps_suppkey = s_suppkey
				and s_nationkey = n_nationkey
				and n_name = 'GERMANY'
		)
order by
	value desc;

\timing
