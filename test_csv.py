import csv


measure_file = open("saved/test_csv.csv", 'a+', encoding='utf-8', newline='')
measure_csv = csv.writer(measure_file)
measure_csv.writerow([f'base', 'aug'])
measure_csv.writerow([1, 2, 3, 4, 5])


