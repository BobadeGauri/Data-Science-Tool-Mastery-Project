-- Example SQL queries for a table named retail_sales.
-- Import data/retail_sales_raw.csv into your SQL tool first.

-- 1. Total business performance
SELECT
    SUM(units_sold) AS total_units,
    SUM(units_sold * unit_price) AS total_revenue,
    SUM(units_sold * (unit_price - unit_cost)) AS total_profit
FROM retail_sales;

-- 2. Revenue and profit by category
SELECT
    category,
    SUM(units_sold) AS units_sold,
    SUM(units_sold * unit_price) AS revenue,
    SUM(units_sold * (unit_price - unit_cost)) AS profit
FROM retail_sales
GROUP BY category
ORDER BY revenue DESC;

-- 3. Monthly revenue trend (syntax shown for SQLite)
SELECT
    strftime('%Y-%m', date) AS month,
    SUM(units_sold * unit_price) AS revenue
FROM retail_sales
GROUP BY strftime('%Y-%m', date)
ORDER BY month;

-- 4. Sales channel comparison
SELECT
    channel,
    COUNT(*) AS transaction_rows,
    SUM(units_sold * unit_price) AS revenue
FROM retail_sales
GROUP BY channel
ORDER BY revenue DESC;
