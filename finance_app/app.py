import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

import streamlit as st

def authenticate():
    with st.sidebar:
        st.title("🔐 Login")
        user = st.text_input("Username")
        pwd = st.text_input("Password", type="password")
        if st.button("Login"):
            if user == st.secrets["auth"]["username"] and pwd == st.secrets["auth"]["password"]:
                st.session_state["auth"] = True
            else:
                st.error("Invalid credentials")
        if not st.session_state.get("auth"):
            st.stop()

authenticate()

DB_PATH = "personal_finance.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT,
        type TEXT,
        category TEXT,
        amount REAL,
        note TEXT
    )''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Budgets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        category TEXT,
        month TEXT,
        budget_amount REAL
    )''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Debts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        creditor TEXT,
        balance REAL,
        interest_rate REAL,
        due_date TEXT,
        min_payment REAL,
        credit_limit REAL,
        initial_balance REAL
    )''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Goals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        description TEXT,
        target_amount REAL,
        current_amount REAL,
        deadline TEXT
    )''')
    conn.commit()
    conn.close()

def add_transaction(date, type_, category, amount, note):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO Transactions (date, type, category, amount, note)
    VALUES (?, ?, ?, ?, ?)
    ''', (date, type_, category, amount, note))
    conn.commit()
    conn.close()

def get_monthly_summary():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query('''
    SELECT 
        strftime('%Y-%m', date) AS month,
        SUM(CASE WHEN type = 'Income' THEN amount ELSE 0 END) AS income,
        SUM(CASE WHEN type = 'Expense' THEN amount ELSE 0 END) AS expenses,
        SUM(CASE WHEN type = 'Debt Payment' THEN amount ELSE 0 END) AS debt,
        SUM(amount) AS net_cash_flow
    FROM Transactions
    GROUP BY month
    ORDER BY month DESC
    ''', conn)
    conn.close()
    return df

def get_budget_status():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query('''
    SELECT B.category, B.month, B.budget_amount, 
           IFNULL(SUM(T.amount), 0) as spent
    FROM Budgets B
    LEFT JOIN Transactions T ON B.category = T.category AND strftime('%Y-%m', T.date) = B.month AND T.type = 'Expense'
    GROUP BY B.category, B.month, B.budget_amount
    ''', conn)
    conn.close()
    df['balance'] = df['budget_amount'] + df['spent']
    df['alert'] = df['balance'] < 0
    return df

def get_transactions():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM Transactions ORDER BY date DESC", conn)
    conn.close()
    return df

def get_debts():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM Debts", conn)
    conn.close()
    return df

def calculate_kpis(df, debts):
    income = df[df['type'] == 'Income']['amount'].sum()
    expenses = df[df['type'] == 'Expense']['amount'].sum()
    debt = df[df['type'] == 'Debt Payment']['amount'].sum()
    savings = income + expenses + debt
    savings_rate = (savings / income) * 100 if income else 0
    expense_ratio = (-expenses / income) * 100 if income else 0

    # Credit utilization
    debts['credit_util'] = debts.apply(lambda x: (x['balance'] / x['credit_limit']) * 100 if x['credit_limit'] > 0 else 0, axis=1)
    avg_credit_util = debts['credit_util'].mean() if not debts.empty else 0

    # Debt reduction
    debts['debt_reduction'] = debts.apply(lambda x: ((x['initial_balance'] - x['balance']) / x['initial_balance']) * 100 if x['initial_balance'] > 0 else 0, axis=1)
    avg_debt_reduction = debts['debt_reduction'].mean() if not debts.empty else 0

    # Emergency fund coverage
    emergency_fund = df[(df['type'] == 'Income') & (df['category'] == 'Emergency Fund')]['amount'].sum()
    avg_monthly_expense = -expenses / len(df['date'].dt.to_period('M').unique()) if income > 0 else 0
    emergency_months = emergency_fund / avg_monthly_expense if avg_monthly_expense > 0 else 0

    return income, -expenses, -debt, savings_rate, expense_ratio, savings, avg_credit_util, avg_debt_reduction, emergency_months

def forecast_trend(df, category):
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['type'] == category].copy()
    if df.empty:
        return [], []
    df = df.groupby(df['date'].dt.to_period('M')).sum().reset_index()
    df['date'] = df['date'].astype(str)
    df['month_index'] = range(len(df))
    X = df[['month_index']]
    y = df['amount']
    model = LinearRegression()
    model.fit(X, y)
    future_index = np.array([[len(df) + i] for i in range(1, 7)])
    predictions = model.predict(future_index)
    return df['date'].tolist(), y.tolist(), [f"+{i}" for i in range(1, 7)], predictions.tolist()

def debt_payoff_simulator(debts, method='snowball'):
    if debts.empty:
        return pd.DataFrame()
    debts = debts.sort_values('balance' if method == 'snowball' else 'interest_rate', ascending=True if method == 'snowball' else False)
    results = []
    for _, row in debts.iterrows():
        balance = row['balance']
        rate = row['interest_rate'] / 100 / 12
        payment = row['min_payment']
        months = 0
        while balance > 0:
            interest = balance * rate
            balance = max(0, balance + interest - payment)
            months += 1
        results.append({"Creditor": row['creditor'], "Months to Payoff": months})
    return pd.DataFrame(results)

def main():
    st.set_page_config("Personal Finance Tracker", layout="wide")
    init_db()
    st.title("📊 Personal Finance Tracker")

    df_transactions = get_transactions()
    df_transactions['date'] = pd.to_datetime(df_transactions['date'])
    df_debts = get_debts()

    income, expenses, debt, savings_rate, expense_ratio, savings, credit_util, debt_reduction, emergency_months = calculate_kpis(df_transactions, df_debts)

    col1, col2, col3 = st.columns(3)
    col1.metric("Savings Rate", f"{savings_rate:.1f}%")
    col2.metric("Credit Utilization", f"{credit_util:.1f}%")
    col3.metric("Debt Reduction", f"{debt_reduction:.1f}%")

    col4, col5 = st.columns(2)
    col4.metric("Emergency Fund Coverage", f"{emergency_months:.1f} months")
    col5.metric("Net Cash Flow", f"R{income + expenses + debt:,.2f}")

    st.subheader("📈 Expense Trend Forecast")
    months, values, future_months, forecasts = forecast_trend(df_transactions, "Expense")
    plt.figure(figsize=(10, 4))
    plt.plot(months, values, label="Actual", marker='o')
    plt.plot(future_months, forecasts, label="Forecast", linestyle='--', marker='x')
    plt.xticks(rotation=45)
    plt.title("Expense Forecast")
    plt.xlabel("Month")
    plt.ylabel("Amount (R)")
    plt.legend()
    st.pyplot(plt)

    st.success("All advanced KPIs are now active: savings, credit, debt, emergency fund.")

if __name__ == '__main__':
    main()
