-- Placeholder cohort extraction query.
-- No real database connection, schema, or patient data is included in this project.
-- Adapt table/column names to your own governed clinical data warehouse.

SELECT
    patient_id,
    admission_id,
    age,
    sex,
    admission_type,
    comorbidity_score,
    vital_signs_summary,
    lab_results_summary,
    in_hospital_mortality
FROM
    placeholder_schema.placeholder_admissions_table
WHERE
    admission_date BETWEEN :start_date AND :end_date;
