\o out.txt
SET random_page_cost = 32;

\timing

select
	l_orderkey,
	l_shipmode,
	sum(case
		when o_orderpriority = '1-URGENT'
			or o_orderpriority = '2-HIGH'
			then 1
		else 0
	end) as high_line_count,
	sum(case
		when o_orderpriority <> '1-URGENT'
			and o_orderpriority <> '2-HIGH'
			then 1
		else 0
	end) as low_line_count
from
	orders,
	lineitem
where
	o_orderkey = l_orderkey
	and o_orderdate < date '1995-03-01'
	and l_shipmode in ('RAIL', 'TRUCK')
	and l_commitdate < l_receiptdate
	and l_receiptdate >= date '1995-03-01'
	and l_receiptdate < date '1995-03-01' + interval '1' year
group by
	l_orderkey,
	l_shipmode
order by
	l_orderkey asc,
	l_shipmode desc

LIMIT 1;

\timing
