from testcase_generation import generate_test_cases
# from few_shot_prompting import get_few_shot_prompting_response as get_few_shot_response
from prompting import get_few_shot_prompting_response as get_few_shot_response
from testcase_generation import *

print("-----------TestCaseGenerator----------")

FRD_input = '''     
Auto_Drift_Estimate - [enabled, not_enabled]
a0 - [1, 2, 1]
a1 - [2, 2, 1]
a2 - [3, 5, 1]
a3 - [0, 2, 1]
a4 - [0, 1, 1]
a5 - [1, 2, 1]
a6 - [0, 2, 1]
t0 - [50, 100, 20]
actual_frequency - [7000, 7010, 3]
present_temperature - [-10, 50, 8]
_RANGE_END_TAG_

If Auto_Drift_Estimate Enabled
If -10 < present_temperature < 55 then
delfbyf0 = a0 + a1 * (present_temperature - temp0) + a2 * (present_temperature - temp0)**2 + a3 * (present_temperature - temp0)**3 + a4 * (present_temperature - temp0)**4 + a5 * (present_temperature - temp0)**5 + a6 * (present_temperature - temp0)**6
Where
f0 = (1/0.000125)
theoretical_frequency = f0
deltaf = actual_frequency - theoretical_frequency
temp0 = 25 
deltaf = f0 * delfbyf0
new_time_period = 1/(f0 + deltaf)
delta_time_period = new_time_period - t0
'''

print("Generating Testcases... Please wait a minute...")

df = generate_test_cases(FRD_input)

print("\n===== FINAL DATAFRAME =====")
print(df.head())



