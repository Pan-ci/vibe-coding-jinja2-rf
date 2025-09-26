from pathlib import Path
import sys

root = Path(__file__).parent.parent.parent.resolve()
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

print('root: ', root)

from app.dependencies.mapping import load_and_clean_data

csv_path = root / 'data' / 'Student Mental health.csv'
df = load_and_clean_data(csv_path)

courses_series = df['What is your course?'].value_counts()
genders_series = df['Choose your gender'].value_counts()
years_series = df['Your current year of Study'].value_counts()
cgpas_series = df['What is your CGPA?'].value_counts()
marital_statuses_series = df['Marital status'].value_counts()

courses = sorted(courses_series.index.to_list())
genders = sorted(genders_series.index.to_list())
years = sorted(years_series.index.to_list())

cgpa_raw = cgpas_series.index.to_list()
cgpa_order = ['0 - 1.99', '2.00 - 2.49', '2.50 - 2.99', '3.00 - 3.49', '3.50 - 4.00']
cgpas = sorted(cgpa_raw, key=lambda x: cgpa_order.index(x) if x in cgpa_order else 999)

marital_statuses = sorted(marital_statuses_series.index.to_list())

'''
# lebih ringkas, jika sudah yakin semuanya tercakup di order tersebut
cgpas = cgpa_order

# untuk string float?
cgpas = sorted(cgpas_series.index.to_list(), key=lambda x: float(x))
'''