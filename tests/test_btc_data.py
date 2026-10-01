"""Offline regression checks for timestamp formats and market-data validation."""
from io import BytesIO
from pathlib import Path
import sys
import unittest
from zipfile import ZipFile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import btc_data as data


class BTCDataTests(unittest.TestCase):
    def test_archive_millisecond_and_microsecond_timestamps(self):
        for date, divisor in [('2020-04-01', 10**6), ('2025-01-01', 10**3)]:
            with self.subTest(date=date):
                start = pd.Timestamp(date, tz='UTC')
                payload = BytesIO()
                with ZipFile(payload, 'w') as archive:
                    archive.writestr('candles.csv', f'{start.value // divisor},100,102,99,101,5,0,0,0,0,0,0\n')
                frame = data.parse_archive(payload.getvalue())
                self.assertEqual(frame.index[0], start)
                self.assertEqual(frame.close.iloc[0], 101)

    def test_validation_rejects_gaps_duplicates_and_invalid_prices(self):
        start = pd.Timestamp('2020-04-01', tz='UTC')
        end = start + pd.Timedelta(hours=3)
        frame = pd.DataFrame(dict(open=[100.]*3, high=[102.]*3, low=[99.]*3,
                                  close=[101.]*3, volume=[5.]*3),
                             index=pd.date_range(start, periods=3, freq='h'))
        pd.testing.assert_frame_equal(data.validate(frame, start, end), frame)
        bad_high = frame.copy()
        bad_high.loc[start, 'high'] = 98
        for invalid in [frame.drop(frame.index[1]), pd.concat([frame, frame.iloc[:1]]), bad_high]:
            with self.assertRaises(ValueError):
                data.validate(invalid, start, end)
        sparse = frame.drop(frame.index[1])
        pd.testing.assert_frame_equal(data.validate(sparse, start, end,
                                      declared_missing=[frame.index[1]]), sparse)


if __name__ == '__main__':
    unittest.main(verbosity=2)
