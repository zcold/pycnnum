""" Chinese number <=> int/float conversion methods """

from __future__ import annotations
from typing import Callable, Tuple, Callable, Union, List

# region chinese chars
from typing import Tuple


CHINESE_DIGITS: str = "零一二三四五六七八九"
"""零一二三四五六七八九

Normal simplified/traditional Chinese charactors for 0123456789.
"""

CAPITAL_CHINESE_DIGITS: str = "零壹贰叁肆伍陆柒捌玖"
"""零壹贰叁肆伍陆柒捌玖

Capitalized Chinese charactors for 0123456789.
Same for both simplified and traditional Chinese.
"""

SMALLER_CHINESE_NUMBERING_UNITS_SIMPLIFIED: str = "十百千万"
"""十百千万

Simplified Chinese charactors for \(10^1\), \(10^2\), \(10^3\), \(10^4\).
Not like other numbering systems,
a Chinese number is grouped by four decimal digits.
"""

SMALLER_CHINESE_NUMBERING_UNITS_TRADITIONAL: str = "拾佰仟萬"
"""拾佰仟萬

Traditional Chinese charactors for \(10^1\), \(10^2\), \(10^3\), \(10^4\).
Also used as capitalized units for simplified Chinese.
Not like other numbering systems,
a Chinese number is grouped by four decimal digits.
"""

LARGER_CHINESE_NUMBERING_UNITS_SIMPLIFIED: str = "亿兆京垓秭穰沟涧正载"
"""亿兆京垓秭穰沟涧正载

    Simplified Chinese charactors for larger units.

    ---

    For \(i \in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]\):

    | numbering_type | value                |
    |----------------|----------------------|
    | `low`          | \(10^{8 + i}\)       |
    | `mid`          | \(10^{8 + i*4}\)     |
    | `high`         | \(10^{8 + 2^{i+3}}\) |

    ---

    |type|亿|兆|京|垓|秭|穰|沟|涧|正|载|
    |---|---|---|---|---|---|---|---|---|---|---|
    |`low`|\(10^{8}\)|\(10^{9}\)|\(10^{10}\)|\(10^{11}\)|\(10^{12}\)|\(10^{13}\)|\(10^{14}\)|\(10^{15}\)|\(10^{16}\)|\(10^{17}\)|
    |`mid`|\(10^{8}\)|\(10^{12}\)|\(10^{16}\)|\(10^{20}\)|\(10^{24}\)|\(10^{28}\)|\(10^{32}\)|\(10^{36}\)|\(10^{40}\)|\(10^{44}\)|
    |`high`|\(10^{8}\)|\(10^{16}\)|\(10^{32}\)|\(10^{64}\)|\(10^{128}\)|\(10^{256}\)|\(10^{512}\)|\(10^{1024}\)|\(10^{2048}\)|\(10^{4096}\)|

"""

LARGER_CHINESE_NUMBERING_UNITS_TRADITIONAL: str = "億兆京垓秭穰溝澗正載"
"""億兆京垓秭穰溝澗正載

    Traditional Chinese charactors for larger units.
    Also used as capitalized units for simplified Chinese.

    ---

    For \(i \in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]\):

    | numbering_type | value                |
    |----------------|----------------------|
    | `low`          | \(10^{8 + i}\)       |
    | `mid`          | \(10^{8 + i*4}\)     |
    | `high`         | \(10^{8 + 2^{i+3}}\) |

    ---

    |type|亿|兆|京|垓|秭|穰|沟|涧|正|载|
    |---|---|---|---|---|---|---|---|---|---|---|
    |`low`|\(10^{8}\)|\(10^{9}\)|\(10^{10}\)|\(10^{11}\)|\(10^{12}\)|\(10^{13}\)|\(10^{14}\)|\(10^{15}\)|\(10^{16}\)|\(10^{17}\)|
    |`mid`|\(10^{8}\)|\(10^{12}\)|\(10^{16}\)|\(10^{20}\)|\(10^{24}\)|\(10^{28}\)|\(10^{32}\)|\(10^{36}\)|\(10^{40}\)|\(10^{44}\)|
    |`high`|\(10^{8}\)|\(10^{16}\)|\(10^{32}\)|\(10^{64}\)|\(10^{128}\)|\(10^{256}\)|\(10^{512}\)|\(10^{1024}\)|\(10^{2048}\)|\(10^{4096}\)|

"""

ZERO_ALT: str = "〇〇"
"""〇〇

Another version of zero in simplified and traditional Chinese
"""

TWO_ALT: str = "两兩"
"""两兩

Another version of two in simplified and traditional Chinese
"""

POSITIVE: str = "正正"
"""正正

Positive in simplified and traditional Chinese
"""

NEGATIVE: str = "负負"
"""负負

negative in simplified and traditional Chinese
"""

POINT: str = "点點"
"""点點

point in simplified and traditional Chinese
"""
# endregion

NUMBERING_TYPES: Tuple[str, str, str] = ("low", "mid", "high")
"""low, mid, high

    Chinese numbering types:

    |type|亿|兆|京|垓|秭|穰|沟|涧|正|载|
    |---|---|---|---|---|---|---|---|---|---|---|
    |`'low'`|\(10^{8}\)|\(10^{9}\)|\(10^{10}\)|\(10^{11}\)|\(10^{12}\)|\(10^{13}\)|\(10^{14}\)|\(10^{15}\)|\(10^{16}\)|\(10^{17}\)|
    |`'mid'`|\(10^{8}\)|\(10^{12}\)|\(10^{16}\)|\(10^{20}\)|\(10^{24}\)|\(10^{28}\)|\(10^{32}\)|\(10^{36}\)|\(10^{40}\)|\(10^{44}\)|
    |`'high'`|\(10^{8}\)|\(10^{16}\)|\(10^{32}\)|\(10^{64}\)|\(10^{128}\)|\(10^{256}\)|\(10^{512}\)|\(10^{1024}\)|\(10^{2048}\)|\(10^{4096}\)|
"""

# region char classes
class ChineseChar:
    """Base Chinese char class.

    Each object has simplified and traditional strings.
    When converted to string, it will shows the simplified string or traditional string or space `' '`.

    Example:
        >>> negative = ChineseChar(simplified="负", traditional="負")
        >>> negative.simplified
        '负'
        >>> negative.traditional
        '負'
    """

    def __init__(self, simplified: str, traditional: str) -> None:
        """
        Args:
            simplified (str): Simplified Chinese char
            traditional (str): Traditional Chinese char
        """
        self.simplified = simplified
        self.traditional = traditional

    def __str__(self) -> str:
        return self.simplified or self.traditional or " "

    def __repr__(self) -> str:
        return self.__str__()


class ChineseNumberUnit(ChineseChar):
    """Chinese number unit class

    Each of it is an `ChineseChar` with additional capitalize type strings.

    Example:
        >>> wan = ChineseNumberUnit(4, "万", "萬", "萬", "萬")
        >>> wan
        10^4
    """

    def __init__(
        self,
        power: int,
        simplified: str,
        traditional: str,
        capital_simplified: str,
        capital_traditional: str,
    ) -> None:
        """
        Args:
            power (int): The power of this unit, e.g. `power` = 4 for `'万'` ( \(10^4\) )
            simplified (str): Charactor in simplified Chinese
            traditional (str): Charactor in traditional Chinese
            capital_simplified (str): Capitalized charactor in simplified Chinese
            capital_traditional (str): Capitalized charactor in traditional Chinese
        """
        super(ChineseNumberUnit, self).__init__(simplified, traditional)
        self.power = power
        self.capital_simplified = capital_simplified
        self.capital_traditional = capital_traditional

    def __str__(self) -> str:
        return f"10^{self.power}"

    @classmethod
    def create(
        cls,
        index: int,
        chars: str,
        numbering_type: str = NUMBERING_TYPES[1],
        small_unit: bool = False,
    ) -> ChineseNumberUnit:
        """Create one unit charactor based on index in units in

        - `SMALLER_CHINESE_NUMBERING_UNITS_SIMPLIFIED`
        - `SMALLER_CHINESE_NUMBERING_UNITS_TRADITIONAL`
        - `LARGER_CHINESE_NUMBERING_UNITS_SIMPLIFIED`
        - `LARGER_CHINESE_NUMBERING_UNITS_TRADITIONAL`


        Args:
            index (int): Zero based index in larger units.
            chars (str): simplified and traditional charactors.
            numbering_type (str, optional): Numbering type. Defaults to `NUMBERING_TYPES[1]`.
            small_unit (bool, optional): the unit is small unit (less than \(10^5\) ). Defaults to False.

        Raises:
            ValueError: Raised when

                - invalid `index` is provided
                - invalid `numbering_type` is provided

        Returns:
            ChineseNumberUnit: Created unit object

        Example:
            >>> wan = ChineseNumberUnit.create(3, "万萬萬萬", small_unit=True)
            >>> wan
            10^4
        """
        if index > len(LARGER_CHINESE_NUMBERING_UNITS_SIMPLIFIED):
            raise ValueError(
                f"{index} should be from 0 to {len(LARGER_CHINESE_NUMBERING_UNITS_SIMPLIFIED)}."
            )

        if small_unit:
            if index > len(SMALLER_CHINESE_NUMBERING_UNITS_SIMPLIFIED):
                raise ValueError(
                    f"{index} should be from 0 to {len(SMALLER_CHINESE_NUMBERING_UNITS_SIMPLIFIED)}."
                )

            return ChineseNumberUnit(
                power=index + 1,
                simplified=chars[0],
                traditional=chars[1],
                capital_simplified=chars[1],
                capital_traditional=chars[1],
            )

        if index > len(LARGER_CHINESE_NUMBERING_UNITS_SIMPLIFIED):
            raise ValueError(
                f"{index} should be from 0 to {len(LARGER_CHINESE_NUMBERING_UNITS_SIMPLIFIED)}."
            )

        if numbering_type == NUMBERING_TYPES[0]:
            return ChineseNumberUnit(
                power=index + 8,
                simplified=chars[0],
                traditional=chars[1],
                capital_simplified=chars[0],
                capital_traditional=chars[1],
            )

        if numbering_type == NUMBERING_TYPES[1]:
            return ChineseNumberUnit(
                power=(index + 2) * 4,
                simplified=chars[0],
                traditional=chars[1],
                capital_simplified=chars[0],
                capital_traditional=chars[1],
            )

        if numbering_type == NUMBERING_TYPES[2]:
            return ChineseNumberUnit(
                power=pow(2, index + 3),
                simplified=chars[0],
                traditional=chars[1],
                capital_simplified=chars[0],
                capital_traditional=chars[1],
            )

        raise ValueError(
            f"Numbering type should be in {NUMBERING_TYPES} but {numbering_type} is provided."
        )


class ChineseNumberDigit(ChineseChar):
    """Chinese number digit class

    Example:
        >>> san = ChineseNumberDigit(3, *"三叁叁叁",)
        >>> san
        3
    """

    def __init__(
        self,
        int_value: int,
        simplified: str,
        traditional: str,
        capital_simplified: str,
        capital_traditional: str,
        alt_s: str = "",
        alt_t: str = "",
    ):
        """
        Args:
            int_value (int): int value of the digit, 0 to 9.
            simplified (str): Charactor in simplified Chinese.
            traditional (str): Charactor in traditional Chinese.
            capital_simplified (str): Capitalized charactor in simplified Chinese.
            capital_traditional (str): Capitalized charactor in traditional Chinese.
            alt_s (str, optional): Alternative simplified charactor. Defaults to "".
            alt_t (str, optional): Alternative traditional charactor. Defaults to "".
        """
        super(ChineseNumberDigit, self).__init__(simplified, traditional)
        self.int_value = int_value
        self.capital_simplified = capital_simplified
        self.capital_traditional = capital_traditional
        self.alt_s = alt_s
        self.alt_t = alt_t

    def __str__(self):
        return str(self.int_value)


class ChineseMath(ChineseChar):
    """
    Chinese math operators

    Example:
        >>> positive = ChineseMath(*'正正+', lambda x: +x)
        >>> positive.symbol
        '+'
    """

    def __init__(
        self,
        simplified: str,
        traditional: str,
        symbol: str,
        expression: Callable = None,
    ):
        """
        Args:
            simplified (str): Simplified charactor.
            traditional (str): Traditional charactor.
            symbol (str): Mathematical symbol, e.g. '+'.
            expression (Callable, optional): Callable for this math operator. Defaults to `None`.
        """
        super(ChineseMath, self).__init__(simplified, traditional)
        self.symbol = symbol
        self.expression = expression
        self.capital_simplified = simplified
        self.capital_traditional = traditional


class MathSymbols:
    """Math symbols used in Chinese for both traditional and simplified Chinese

    - positive = ['正', '正']
    - negative = ['负', '負']
    - point = ['点', '點']

    Used in `NumberingSystem`.
    """

    def __init__(
        self, positive: ChineseMath, negative: ChineseMath, point: ChineseMath
    ):
        """
        Args:
            positive (ChineseMath): Positive
            negative (ChineseMath): Negative
            point (ChineseMath): Decimal point
        """
        self.positive = positive
        self.negative = negative
        self.point = point

    def __iter__(self):
        for v in self.__dict__.values():
            yield v


class NumberingSystem:
    """Numbering system class"""

    def __init__(self, numbering_type: str = NUMBERING_TYPES[1]) -> None:
        """
        Args:
            numbering_type (str, optional): Numbering type. Defaults to `NUMBERING_TYPES[1]`.

        Example:
            >>> low = NumberingSystem('low')
            >>> low.digits
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            >>> low.units
            [10^1, 10^2, 10^3, 10^4, 10^8, 10^9, 10^10, 10^11, 10^12, 10^13, 10^14, 10^15, 10^16, 10^17]
            >>> mid = NumberingSystem('mid')
            >>> mid.units
            [10^1, 10^2, 10^3, 10^4, 10^8, 10^12, 10^16, 10^20, 10^24, 10^28, 10^32, 10^36, 10^40, 10^44]
            >>> high = NumberingSystem('high')
            >>> high.units
            [10^1, 10^2, 10^3, 10^4, 10^8, 10^16, 10^32, 10^64, 10^128, 10^256, 10^512, 10^1024, 10^2048, 10^4096]
        """

        # region units of '亿' and larger
        all_larger_units = zip(
            LARGER_CHINESE_NUMBERING_UNITS_SIMPLIFIED,
            LARGER_CHINESE_NUMBERING_UNITS_TRADITIONAL,
        )
        larger_units = [
            ChineseNumberUnit.create(
                index=index,
                chars=simplified + traditional,
                numbering_type=numbering_type,
                small_unit=False,
            )
            for index, (simplified, traditional) in enumerate(all_larger_units)
        ]
        # endregion

        # region units of '十, 百, 千, 万'
        all_smaller_units = zip(
            SMALLER_CHINESE_NUMBERING_UNITS_SIMPLIFIED,
            SMALLER_CHINESE_NUMBERING_UNITS_TRADITIONAL,
        )
        smaller_units = [
            ChineseNumberUnit.create(
                index=index,
                chars=simplified + traditional,
                numbering_type=numbering_type,
                small_unit=True,
            )
            for index, (simplified, traditional) in enumerate(all_smaller_units)
        ]
        # endregion

        # region digits
        chinese_digits = zip(
            CHINESE_DIGITS,
            CHINESE_DIGITS,
            CAPITAL_CHINESE_DIGITS,
            CAPITAL_CHINESE_DIGITS,
        )
        digits = [ChineseNumberDigit(i, *v) for i, v in enumerate(chinese_digits)]
        digits[0].alt_s, digits[0].alt_t = ZERO_ALT, ZERO_ALT
        digits[2].alt_s, digits[2].alt_t = TWO_ALT[0], TWO_ALT[1]
        # endregion

        # region math operators
        positive_cn = ChineseMath(
            simplified=POSITIVE[0],
            traditional=POSITIVE[1],
            symbol="+",
            expression=lambda x: x,
        )
        negative_cn = ChineseMath(
            simplified=NEGATIVE[0],
            traditional=NEGATIVE[1],
            symbol="-",
            expression=lambda x: -x,
        )
        point_cn = ChineseMath(
            simplified=POINT[0],
            traditional=POINT[1],
            symbol=".",
            expression=lambda integer, decimal: float(
                str(integer) + "." + str(decimal)
            ),
        )
        # endregion

        self.units = smaller_units + larger_units
        self.digits = digits
        self.math = MathSymbols(positive_cn, negative_cn, point_cn)


# endregion char classes

SymbolType = Union[ChineseNumberUnit, ChineseNumberDigit, ChineseMath]


def cn2num(
    chinese_string: str, numbering_type: str = NUMBERING_TYPES[1]
) -> Union[int, float]:
    """Convert Chinese number to `int` or `float` value。

    Args:
        chinese_string (str): Chinese number
        numbering_type (str, optional): numbering type. Defaults to `NUMBERING_TYPES[1]`.

    Raises:
        ValueError: Raised when

            - a charactor is not in the numbering system, e.g. '你' is not a number nor a unit

    Returns:
        Union[int, float]: `int` or `float` value

    Example:
        >>> cn2num("一百八")
        180
        >>> cn2num("一百八十")
        180
        >>> cn2num("一百八点五六七")
        180.567
        >>> cn2num("两千万一百八十")
        20000180
    """

    def get_symbol(char: str, system: NumberingSystem) -> SymbolType:
        """Get symbol based on charactor

        Args:
            char (str): One charactor
            system (NumberingSystem): Numbering system

        Raises:
            ValueError: a charactor is not in the numbering system, e.g. '你' is not a number nor a unit

        Returns:
            SymbolType: unit, digit or math operator
        """
        for u in system.units:
            if char in [
                u.traditional,
                u.simplified,
                u.capital_simplified,
                u.capital_traditional,
            ]:
                return u
        for d in system.digits:
            if char in [
                d.traditional,
                d.simplified,
                d.capital_simplified,
                d.capital_traditional,
                d.alt_s,
                d.alt_t,
            ]:
                return d
        for m in system.math:
            if char in [m.traditional, m.simplified]:
                return m
        raise ValueError(f"{char} is not in system.")

    def string2symbols(
        chinese_string: str, system: NumberingSystem
    ) -> Tuple[List[SymbolType], List[SymbolType]]:
        """String to symbols

        Args:
            chinese_string (str): Chinese number
            system (NumberingSystem): Numbering system

        Returns:
            Tuple[List[SymbolType], List[SymbolType]]: Integer symbols, decimal symbols
        """
        int_string, dec_string = chinese_string, ""
        for p in [system.math.point.simplified, system.math.point.traditional]:
            if p not in chinese_string:
                continue
            int_string, dec_string = chinese_string.split(p)
            break
        integer_value = [get_symbol(c, system) for c in int_string]
        decimal_value = [get_symbol(c, system) for c in dec_string]
        return integer_value, decimal_value

    def refine_symbols(
        integer_symbols: List[SymbolType], system: NumberingSystem
    ) -> List[SymbolType]:
        """Refine symbols

        Example:
            - `一百八` == `一百八十`
            - `一亿一千三百万` to `一亿 一千万 三百万`
            - `一万四` == `一万四千`
            - ·两千万· == `两 10^7`

        Args:
            integer_symbols (List[SymbolType]): Raw integer symbols
            system (NumberingSystem): Numbering system

        Returns:
            List[SymbolType]: Refined symbols


        """
        if not integer_symbols:
            return integer_symbols

        # First symbol is unit, e.g. "十五"
        if (
            isinstance(integer_symbols[0], ChineseNumberUnit)
            and integer_symbols[0].power == 1
        ):
            integer_symbols = [system.digits[1]] + integer_symbols  # type: ignore

        # last symbol is digit and the second last symbol is unit, e.g. "十五"
        if len(integer_symbols) > 1:
            if isinstance(integer_symbols[-1], ChineseNumberDigit) and isinstance(
                integer_symbols[-2], ChineseNumberUnit
            ):
                # add a dummy unit
                integer_symbols += [
                    ChineseNumberUnit(integer_symbols[-2].power - 1, "", "", "", "")
                ]

        result: List[SymbolType] = list()
        unit_count = 0
        for s in integer_symbols:
            if isinstance(s, ChineseNumberDigit):
                result.append(s)
                unit_count = 0
                continue

            if not isinstance(s, ChineseNumberUnit):
                continue

            current_unit = ChineseNumberUnit(s.power, "", "", "", "")
            unit_count += 1

            # store the first met unit
            if unit_count == 1:
                result.append(current_unit)
                continue

            # if there are more than one units, e.g. "两千万"
            if unit_count > 1:
                for i in range(len(result)):
                    if not isinstance(result[-i - 1], ChineseNumberUnit):
                        continue
                    if result[-i - 1].power < current_unit.power:  # type: ignore
                        result[-i - 1] = ChineseNumberUnit(
                            result[-i - 1].power + current_unit.power, "", "", "", ""  # type: ignore
                        )
        return result

    def compute_value(integer_symbols: List[SymbolType]) -> int:
        """Compute the value from symbol

        When current unit is larger than previous unit, current unit * all previous units will be used as all previous units.
        e.g. '两千万' = 2000 * 10000 not 2000 + 10000

        Args:
            integer_symbols (List[SymbolType]): Symbols, without point

        Returns:
            int: value
        """
        value = [0]
        last_power = 0
        for s in integer_symbols:
            if isinstance(s, ChineseNumberDigit):
                value[-1] = s.int_value
            elif isinstance(s, ChineseNumberUnit):
                value[-1] *= pow(10, s.power)
                if s.power > last_power:
                    value[:-1] = list(map(lambda v: v * pow(10, s.power), value[:-1]))  # type: ignore
                    last_power = s.power
                value.append(0)
        return sum(value)

    system = NumberingSystem(numbering_type)
    int_part, dec_part = string2symbols(chinese_string, system)
    int_part = refine_symbols(int_part, system)
    int_value = compute_value(int_part)
    # skip unit in decimal value
    dec_str = "".join(
        [str(d.int_value) for d in dec_part if isinstance(d, ChineseNumberDigit)]
    )

    if dec_part:
        return float(f"{int_value}.{dec_str}")
    return int_value


def num2cn(
    num: Tuple[int, float, str],
    numbering_type: str = NUMBERING_TYPES[1],
    capitalize: bool = False,
    traditional: bool = False,
    alt_zero: bool = False,
    alt_two: bool = False,
    keep_zeros: bool = True,
) -> str:
    """Integer or float value to Chinese string

    Args:
        num (Tuple[int, float, str]): `int`, `float` or `str` value
        numbering_type (str, optional): Numbering type. Defaults to `NUMBERING_TYPES[1]`.
        capitalize (bool, optional): Capitalized numbers. Defaults to `False`.
        traditional (bool, optional): Traditional Chinese. Defaults to `False`.
        alt_zero (bool, optional): Use alternative form of zero. Defaults to `False`.
        alt_two (bool, optional): Use alternative form of two. Defaults to `False`.
        keep_zeros (bool, optional): Keep Chinese zeros in `num`. Defaults to `True`.

    Example:
        >>> num2cn('023232.005184132423423423300', numbering_type="high", alt_two=True, capitalize=False, traditional=True)
        '零兩萬三仟兩佰三拾二點零零五一八四一三二四二三四二三四二三三'
        >>> num2cn('023232.005184132423423423300', numbering_type="high", alt_two=False, capitalize=False, traditional=True)
        '零二萬三仟二佰三拾二點零零五一八四一三二四二三四二三四二三三'
        >>> num2cn(111180000)
        '一亿一千一百一十八万'
        >>> num2cn(1821010)
        '一百八十二万一千零一十'
        >>> num2cn(182.1)
        '一百八十二点一'
        >>> num2cn('3.4')
        '三点四'
        >>> num2cn(16)
        '十六'
        >>> num2cn(10600)
        '一万零六百'
        >>> num2cn(110)
        '一百一'
        >>> num2cn(1600)
        '一千六'

    """

    def get_value(value_string: str, keep_zeros: bool = True) -> List[SymbolType]:
        """Recursively get values of the number

        Args:
            value_string (str): Value string, e.g. "0.1", "34"
            keep_zeros (bool, optional): Use Chinese zero for leading zeros in `num`. Defaults to `True`.

        Returns:
            List[SymbolType]: List of values
        """

        striped_string = value_string.lstrip("0")

        # record nothing if all zeros
        if not striped_string:
            return []

        # record one digits
        if len(striped_string) == 1:
            if keep_zeros and (len(value_string) != len(striped_string)):
                return [  # type: ignore
                    system.digits[0]
                    for _ in range(len(value_string) - len(striped_string))
                ] + [system.digits[int(striped_string)]]
            return [system.digits[int(striped_string)]]

        # recursively record multiple digits
        result_unit = next(
            u for u in reversed(system.units) if u.power < len(striped_string)
        )
        result_string = value_string[: -result_unit.power]
        return (
            get_value(result_string, keep_zeros)
            + [result_unit]
            + get_value(striped_string[-result_unit.power :], keep_zeros)
        )

    system = NumberingSystem(numbering_type)

    num_str = str(num)
    int_string = num_str
    dec_string = ""

    if "." in num_str:
        int_string, dec_string = num_str.split(".", 1)
        dec_string = dec_string.rstrip("0")

    result_symbols = get_value(int_string, keep_zeros)
    dec_symbols = [system.digits[int(c)] for c in dec_string]

    if "." in num_str:
        result_symbols += [system.math.point] + dec_symbols  # type: ignore

    if alt_two:
        liang = ChineseNumberDigit(
            2,
            system.digits[2].alt_s,
            system.digits[2].alt_t,
            system.digits[2].capital_simplified,
            system.digits[2].capital_traditional,
        )
        for i, v in enumerate(result_symbols):
            if not isinstance(v, ChineseNumberDigit):
                continue
            if v.int_value != 2:
                continue
            next_symbol = result_symbols[i + 1] if i < len(result_symbols) - 1 else None
            previous_symbol = result_symbols[i - 1] if i > 0 else None
            if not isinstance(next_symbol, ChineseNumberUnit):
                continue
            leading_zero = getattr(previous_symbol, "int_value", False) == 0
            previous_is_unit = isinstance(previous_symbol, ChineseNumberUnit)
            if not (leading_zero or previous_is_unit or (previous_symbol is None)):
                continue
            if next_symbol.power > 1:
                result_symbols[i] = liang

    # if capitalize is True, '两' will not be used and `alt_two` has no impact on output
    if traditional:
        attr_name = "traditional"
    else:
        attr_name = "simplified"

    if capitalize:
        attr_name = "capital_" + attr_name

    # remove leading '一' for '十', e.g. 一十六 to 十六
    if (result_symbols[0] == system.digits[1]) and (
        getattr(result_symbols[1], "power", None) == 1
    ):
        result_symbols = result_symbols[1:]

    # remove trailing units, 1600 -> 一千六, 10600 -> 一萬零六百, 101600 -> 十萬一千六
    if len(result_symbols) > 3 and isinstance(result_symbols[-1], ChineseNumberUnit):
        if getattr(result_symbols[::-1][2], "power", None) == (
            result_symbols[-1].power + 1
        ):
            result_symbols = result_symbols[:-1]

    result = "".join([getattr(s, attr_name) for s in result_symbols])

    if alt_zero:
        result = result.replace(
            getattr(system.digits[0], attr_name), system.digits[0].alt_s
        )

    for p in POINT:
        if result.startswith(p):
            return CHINESE_DIGITS[0] + result
    return result


__all__ = ["cn2num", "num2cn"]