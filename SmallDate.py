

class SmallDate:
    def __init__(self, year, month):
        self.year = year
        self.month = month
        assert month in range(1,13), 'Month must ve valid number'

    def smaller_than_or_equal_to(self, other_date):
        if self.year > other_date.year:
            return False
        elif self.year < other_date.year:
            return True
        else:
            return self.month<=other_date.month

    def increase_month(self,inc=1):
        self.month=(self.month+inc)
        while self.month > 12:
            self.month-=12
            self.year+=1

    def __str__(self, separator = ', '): # separator = ', ' for loading data
        year_string = str(self.year)
        month_string = str(self.month)
        if self.month<10:
            month_string = '0'+month_string

        return year_string+separator+ month_string

    def minus(self, monthDiff):
        current_year = self.year
        current_month = self.month
        current_month-=monthDiff
        while current_month < 1:
            current_year-=1
            current_month+=12
        return SmallDate(current_year,current_month)

    def reduce_month(self, red=1):
        self.month=(self.month-red)
        while self.month < 1:
            self.month += 12
            self.year -=1

    def toTupple(self):
        return self.year, self.month