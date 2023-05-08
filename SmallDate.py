

class SmallDate:
    def __init__(self, year, month):
        self.year = year
        self.month = month

    def smaller_than_or_equal_to(self, other_date):
        if self.year > other_date.year:
            return False
        elif self.year < other_date.year:
            return True
        else:
            return self.month<=other_date.month

    def increase_month(self,inc=1):
        self.month=(self.month+inc)
        if self.month > 12:
            self.month-=12
            self.year+=1

    def __str__(self):
        year_string = str(self.year)
        month_string = str(self.month)
        if self.month<10:
            month_string = '0'+month_string

        return year_string+', '+ month_string