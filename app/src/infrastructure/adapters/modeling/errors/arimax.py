
class ConstantInExogAndSpecification(Exception):
    def __str__(self):
        return 'Модель включает в себя константу для тренда, однако экзогенные переменные также включают константный ряд'

