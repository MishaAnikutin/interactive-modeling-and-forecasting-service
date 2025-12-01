# fixme

class StudentTestError(Exception):
    """Базовое исключение для теста Стьюдента"""
    pass


class InvalidDateError(StudentTestError):
    """Ошибка невалидной даты"""
    pass


class InsufficientDataError(StudentTestError):
    """Недостаточно данных для теста"""
    pass
