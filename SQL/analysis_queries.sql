
--1
Alter table dbo.loan
DROP COLUMN id,member_id; 

ALTER TABLE dbo.loan
ADD Loan_ID INT Identity(1,1) NOT NULL;



SELECT * ,
    CASE 
        WHEN loan_status LIKE '%Charged Off%' OR loan_status = 'Default' THEN 1 
        ELSE 0 
    END AS predictor
INTO dbo.loan_model_ready
FROM dbo.loan
WHERE loan_status IN ('Fully Paid', 'Charged Off', 'Default', 
                     'Does not meet the credit policy. Status:Fully Paid',
                     'Does not meet the credit policy. Status:Charged Off');


--2.1
SELECT DISTINCT url
from dbo.loan_model_ready; 


SELECT DISTINCT [desc]
from dbo.loan_model_ready; 

alter table dbo.loan_model_ready
drop column [url],[desc];


--2.2
Alter table dbo.loan_model_ready
add months_sincefrst_credit INT;


UPDATE dbo.loan_model_ready
set months_sincefrst_credit = DAtediff(month, earliest_cr_line,issue_d);

alter table dbo.loan_model_ready
drop column issue_d,earliest_cr_line;



---2.3
SELECT policy_code, COUNT(*) as Total_Count
FROM dbo.loan_model_ready
GROUP BY policy_code;


ALTER table dbo.loan_model_ready
drop column policy_code, emp_title, zip_code



--2.4
SELECT pub_rec, count(*) as number_of_records, AVG(CAST(predictor as float)) as default_rate
from dbo.loan_model_ready
group by pub_rec
order by pub_rec;

alter table dbo.loan_model_ready
add public_record INT;



update dbo.loan_model_ready
set public_record = case
when pub_rec > 0 then 1 
else 0
end;


--2.5
SELECT open_acc, count(*) as number_of_accounts, AVG(CAST(predictor as float)) as default_rate
from dbo.loan_model_ready
group by open_acc
order by open_acc;


update dbo.loan_model_ready
set total_acc= case when total_rec>50 then 50
else total_acc
end;


--2.6
SELECT 
inq_last_6mths, 
COUNT(*) AS number_of_borrowers, 
AVG(CAST(predictor AS FLOAT)) AS default_rate
FROM dbo.loan_model_ready
GROUP BY inq_last_6mths
ORDER BY inq_last_6mths;

update db.loan_model_ready
set inq_last_6mths = case when inq_last_6mths > 6 then 6
when inq_last_6mths is NULL then 0
else inq_last_6mths
end;


--2.7
SELECT delinq_2yrs, count(*) as number_of_accounts, AVG(CAST(predictor as float)) as default_rate
from dbo.loan_model_ready
group by delinq_2yrs
order by delinq_2yrs;

UPDATE dbo.loan_model_ready
SET delinq_2yrs = CASE 
    WHEN delinq_2yrs > 5 THEN 5 
    WHEN delinq_2yrs IS NULL THEN 0 -- Handling those 29 NULLs
    ELSE delinq_2yrs 
END;


UPDATE dbo.loan_model_ready
SET mths_since_last_delinq = COALESCE(mths_since_last_delinq, 999),
 mths_since_last_record = COALESCE(mths_since_last_record, 999);


--2.8
alter table dbo.loan_model_ready
drop column last_pymnt_d,last_pymnt_amnt,next_pymnt_d;



--2.9
update dbo.loan_model_ready
set initial_list_status = case when initial_list_status = 'w' then '1'
else '0'
end;

ALTER TABLE dbo.loan_model_ready 
ALTER COLUMN initial_list_status INT;



--2.10

SELECT 
    ROUND(dti, 0) as dti_rounded, 
    COUNT(*) as num_borrowers, 
    AVG(CAST(predictor AS FLOAT)) as default_rate
FROM dbo.loan_model_ready
GROUP BY ROUND(dti, 0)
ORDER BY dti_rounded;



update dbo.loan_model_ready
set dti = case 
when dti > 40 then 40
when dti<0 or dti is NULL then 19
else dti
end;



--2.11

ALTER TABLE dbo.loan_model_ready ADD is_consolidation INT;

UPDATE dbo.loan_model_ready
SET is_consolidation = CASE 
    WHEN LOWER(title) LIKE '%consol%' OR LOWER(title) LIKE '%card%' THEN 1 
    ELSE 0 
END;

ALTER TABLE dbo.loan_model_ready DROP COLUMN title;
