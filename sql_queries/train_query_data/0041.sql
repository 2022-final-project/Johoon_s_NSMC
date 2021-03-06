\o out.txt
SET random_page_cost = 32;

\timing

select
	nation,
	sum(case
		when nation in ('ETHIOPIA', 'VIETNAM', 'CANADA') then volume
		else 0
	end) / sum(volume) as mkt_share
from
	(
		select
			l_extendedprice * (1 - l_discount) as volume,
			n2.n_name as nation
		from
			part,
			supplier,
			lineitem,
			orders,
			customer,
			nation n1,
			nation n2,
			region
		where
			p_partkey = l_partkey
			and s_suppkey = l_suppkey
			and l_orderkey = o_orderkey
			and o_custkey = c_custkey
			and c_nationkey = n1.n_nationkey
			and n1.n_regionkey = r_regionkey
			and r_name = 'ASIA'
			and s_nationkey = n2.n_nationkey
			and o_orderdate between date '1995-01-01' and date '1996-12-31'
			and p_type = 'STANDARD BRUSHED TIN'
	) as all_nations
group by
	nation
order by
	nation desc;

\timing
