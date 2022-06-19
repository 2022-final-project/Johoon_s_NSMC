\o out11.txt
SET random_page_cost = 32;
\timing

select
	ps_partkey,
	sum(ps_supplycost * ps_availqty) as value
from
	partsupp,
	supplier
where
	ps_suppkey = s_suppkey
group by
	ps_partkey having
		sum(ps_supplycost * ps_availqty) > (
			select
				sum(ps_supplycost * ps_availqty) * 0.00004
			from
				partsupp,
				supplier
			where
				ps_suppkey = s_suppkey
		)
order by
	value asc;
\timing
