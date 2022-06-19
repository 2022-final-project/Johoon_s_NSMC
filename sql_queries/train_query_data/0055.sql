\o out2.txt
SET random_page_cost = 1;
\timing
select
	s_acctbal,
	s_name,
	n_name,
	p_partkey,
	p_mfgr,
	s_address,
	s_phone,
	s_comment
from
	part,
	supplier,
	partsupp,
	nation,
	lineitem
where
	p_partkey = ps_partkey
	and s_suppkey = ps_suppkey
	and ps_partkey = l_partkey
	and ps_suppkey = l_suppkey
	and l_shipdate < date '1994-01-01'
	and p_size = 1
	and p_type like '%STEEL'
	and s_nationkey = n_nationkey
	and p_size between 10 and 26
	and ps_supplycost = (
		select
			min(ps_supplycost)
		from
			partsupp,
			supplier,
			region
		where
			p_partkey = ps_partkey
			and s_suppkey = ps_suppkey
			and r_name = 'ASIA'
	)
order by
	s_acctbal desc,
	n_name asc,
	s_name asc,
	p_partkey desc;

\timing
