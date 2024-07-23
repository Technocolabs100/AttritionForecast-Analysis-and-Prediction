#  Employee Attrition Analysis and Prediction

This report aims to provide insights into the attrition rates within the organization and to identify potential causes of attrition using data from the Power BI dashboard. The analysis focuses on understanding patterns, trends, and correlations related to employee turnover, which is crucial for HR to formulate strategies aimed at retention and organizational stability.

# Steps Involved : - 

# Data Cleaning:- 
The downloaded data contains 35 columns with columns. 
Firstly columns "DailyRate", "HourlyRate", "MonthlyRate" were removed as these columns were not contributing in dasboard creation.
The quality of data is checked to check the null and duplicate values in the table.
Data Types of columns were also checked 
Columns like "StandardHours", "Over18" is removed as it contains non-filterable data(only consist one value)

# Data Manipulation:-

⦁	"target_att"column is created which contains binary boolean value of target column "Attrition"
			formula - target_att = IF('hrdata'[Attrition]="Yes",1,0)
⦁	"age_range" column is created to make "Age" column into a categorical one.
		formula - age_range = 
                           SWITCH (TRUE(),
                           hrdata[Age]>10 && hrdata[Age] <=30, "<30",
                           hrdata[Age]>30 && hrdata[Age] <=40, "30-40",
                           hrdata[Age]>40 && hrdata[Age] <=50,"40-50",
                           hrdata[Age]>50,"above 50")
⦁	"income_slab" column is created to make "MonthlyIncome' column into a categorical one.
		formula - income_slab = 
                              SWITCH (TRUE(),
                              hrdata[MonthlyIncome]>0 && hrdata[MonthlyIncome]<5000, "0-5000",
                              hrdata[MonthlyIncome]>=5000 && hrdata[MonthlyIncome]<10000, "5000-10000",
                              hrdata[MonthlyIncome]>=10000 && hrdata[MonthlyIncome]<15000, "10000-15000",
                              hrdata[MonthlyIncome]>=15000 && hrdata[MonthlyIncome]<20000, "15000-20000")
                              
# Measures created:-
               
               ⦁	Active employess:- 
               	m_active_emp = count(hrdata[EmployeeNumber]) - sum(hrdata[target_att])
               ⦁	Attrition Rate:- 
               		m_attrition_rate = DIVIDE(SUM('hrdata'[target_att]),COUNT('hrdata'[EmployeeNumber]),"")

# FINDINGS:-

Almost every required details can be filtered with the below dashboard.
 ![dasboard](https://github.com/user-attachments/assets/867dcb34-77ef-4689-900c-d7142d8391e4)

From the above dashboard, it's clear that, with increase in salary(monthly income), the attrition rate decreases.
The R&D sector department has more no. of attritions followed by Sales and Human Resource.
Employee with lowest job satisfaction and working in Sales role and Human Resource has highest attrition rate of 58.33% and 50% respectively.
Male Employees have higher attrition rates than Female Employees
Single Males with frequent business travel have high attrition rate, which signifies that people are moving from Sales Job.
Employees with a doctorate education background have the lowest attrition rate.
Single Employees with bachelor's degree having Sales job role are most likely to leave company in 2-3 years as shown in below fig.
 ![bachelor_sales](https://github.com/user-attachments/assets/3a577410-94b0-43cd-be49-358a1ac7dbf3)

Worklife balance, Work Culture also plays a role in attrition_rate. Significant increase in attrition rate are observed with decrease in Worklife balance and Work culture.

# Cause of Attrition

There could be multiple reasons for high attrition rates, especially low monthly income, less hike in salary, motive to have better growth, Worklife balance, job satisfaction level etc.
Through dashboard its clear that attrition rate is increasing because of following parameters:-
		⦁	Employee with bachelor's degree and above are not interested in Sales Job. Job satisfaction level is very low in these Roles.
		⦁	Unmarried employees are more likely to leave company if they are not satisfied with work culture and worklife balance.
		⦁	HR employees are leaving with company for better salary and work culture.
		⦁	Employees having office 12km are more likely to switch the company.
