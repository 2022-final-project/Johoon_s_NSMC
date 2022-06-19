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
	nation
where
	p_partkey = ps_partkey
	and s_suppkey = ps_suppkey
	and p_size = 1
	and p_type like '%STEEL'
	and s_nationkey = n_nationkey
	and ps_availqty < 5999
	and ps_supplycost = (
		select
			min(ps_supplycost)
		from
			partsupp,
			supplier,
			nation,
			region
		where
			p_partkey = ps_partkey
			and s_suppkey = ps_suppkey
			and s_nationkey = n_nationkey
			and n_regionkey = r_regionkey
			and ps_availqty < 5999
			and r_name = 'ASIA'
	)
order by
	s_acctbal desc,
	n_name asc,
	s_name asc,
	p_partkey desc;

\timing
