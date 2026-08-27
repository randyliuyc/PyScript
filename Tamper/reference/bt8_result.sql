--LINKPYTHON_RESULT:
DECLARE @MFGNUM VARCHAR(30) = 'TEST';
DECLARE @MFBLIN INT = 10;
DECLARE @YARNGRP VARCHAR(30) = '';

DECLARE @JSON1 VARCHAR(MAX) = '{
  "error": "运行超时或未找到有效解",
  "results": [],
  "runtime": 2.5549020767211914
}';

DECLARE @JSON1 VARCHAR(MAX) = '{
  "results": [
    {
      "X1": 1.1,
      "X2": 1.11,
      "X3": 1.47,
      "X4": 2.93,
      "cum_error": "0.000331",
      "total_feed_speed_D": 6.073759,
      "stage_label": "1.1-2.5/3.5",
      "assign": [
        {
          "bucket": "A",
          "color": 80,
          "colordes": "WJ 白棉 VG055M ",
          "colorsho": "WJ 白棉",
          "x": 1.1,
          "speed": 0.909091
        },
        {
          "bucket": "B",
          "color": 62,
          "colordes": "SWP本白 VG010M ",
          "colorsho": "SWP本白",
          "x": 2.93,
          "speed": 0.341297
        },
        {
          "bucket": "C",
          "color": 62,
          "colordes": "SWP本白 VG010M ",
          "colorsho": "SWP本白",
          "x": 1.0,
          "speed": 1.0
        },
        {
          "bucket": "D",
          "color": 81,
          "colordes": "W 白棉 VG055M ",
          "colorsho": "W 白棉",
          "x": 1.0,
          "speed": 1.0
        },
        {
          "bucket": "E",
          "color": 62,
          "colordes": "SWP本白 VG010M ",
          "colorsho": "SWP本白",
          "x": 1.11,
          "speed": 0.900901
        },
        {
          "bucket": "F",
          "color": 80,
          "colordes": "WJ 白棉 VG055M ",
          "colorsho": "WJ 白棉",
          "x": 1.11,
          "speed": 0.900901
        },
        {
          "bucket": "G",
          "color": 81,
          "colordes": "W 白棉 VG055M ",
          "colorsho": "W 白棉",
          "x": 2.93,
          "speed": 0.341297
        },
        {
          "bucket": "H",
          "color": 70,
          "colordes": "BC02W VE001M-U-001 ",
          "colorsho": "BC02W",
          "x": 1.47,
          "speed": 0.680272
        }
      ],
      "colors": [
        {
          "color": 62,
          "colordes": "BC02W VE001M-U-001 ",
          "colorsho": "BC02W",
          "target": 36.9,
          "final": 36.92,
          "error": "0.0161"
        },
        {
          "color": 70,
          "colordes": "BC02W VE001M-U-001 ",
          "colorsho": "BC02W",
          "target": 11.2,
          "final": 11.2,
          "error": "0.0002"
        },
        {
          "color": 80,
          "colordes": "BC02W VE001M-U-001 ",
          "colorsho": "BC02W",
          "target": 29.8,
          "final": 29.8,
          "error": "0.0002"
        },
        {
          "color": 81,
          "colordes": "BC02W VE001M-U-001 ",
          "colorsho": "BC02W",
          "target": 22.1,
          "final": 22.08,
          "error": "0.0165"
        }
      ]
    },
    {
      "X1": 1.13,
      "X2": 1.12,
      "X3": 1.49,
      "X4": 3.08,
      "cum_error": "0.000370",
      "total_feed_speed_D": 5.991162,
      "stage_label": "1.1-2.5/3.5",
      "assign": [
        {
          "bucket": "A",
          "color": 62,
          "colordes": "SWP本白 VG010M ",
          "colorsho": "SWP本白",
          "x": 1.13,
          "speed": 0.884956
        },
        {
          "bucket": "B",
          "color": 62,
          "colordes": "SWP本白 VG010M ",
          "colorsho": "SWP本白",
          "x": 3.08,
          "speed": 0.324675
        },
        {
          "bucket": "C",
          "color": 62,
          "colordes": "SWP本白 VG010M ",
          "colorsho": "SWP本白",
          "x": 1.0,
          "speed": 1.0
        },
        {
          "bucket": "D",
          "color": 81,
          "colordes": "W 白棉 VG055M ",
          "colorsho": "W 白棉",
          "x": 1.0,
          "speed": 1.0
        },
        {
          "bucket": "E",
          "color": 80,
          "colordes": "WJ 白棉 VG055M ",
          "colorsho": "WJ 白棉",
          "x": 1.12,
          "speed": 0.892857
        },
        {
          "bucket": "F",
          "color": 80,
          "colordes": "WJ 白棉 VG055M ",
          "colorsho": "WJ 白棉",
          "x": 1.12,
          "speed": 0.892857
        },
        {
          "bucket": "G",
          "color": 81,
          "colordes": "W 白棉 VG055M ",
          "colorsho": "W 白棉",
          "x": 3.08,
          "speed": 0.324675
        },
        {
          "bucket": "H",
          "color": 70,
          "colordes": "BC02W VE001M-U-001 ",
          "colorsho": "BC02W",
          "x": 1.49,
          "speed": 0.671141
        }
      ],
      "colors": [
        {
          "color": 62,
          "colordes": "BC02W VE001M-U-001 ",
          "colorsho": "BC02W",
          "target": 36.9,
          "final": 36.88,
          "error": "0.0185"
        },
        {
          "color": 70,
          "colordes": "BC02W VE001M-U-001 ",
          "colorsho": "BC02W",
          "target": 11.2,
          "final": 11.2,
          "error": "0.0022"
        },
        {
          "color": 80,
          "colordes": "BC02W VE001M-U-001 ",
          "colorsho": "BC02W",
          "target": 29.8,
          "final": 29.81,
          "error": "0.0058"
        },
        {
          "color": 81,
          "colordes": "BC02W VE001M-U-001 ",
          "colorsho": "BC02W",
          "target": 22.1,
          "final": 22.11,
          "error": "0.0105"
        }
      ]
    },
    {
      "X1": 1.11,
      "X2": 1.13,
      "X3": 1.49,
      "X4": 3.08,
      "cum_error": "0.000387",
      "total_feed_speed_D": 5.991304,
      "stage_label": "1.1-2.5/3.5",
      "assign": [
        {
          "bucket": "A",
          "color": 80,
          "colordes": "WJ 白棉 VG055M ",
          "colorsho": "WJ 白棉",
          "x": 1.11,
          "speed": 0.900901
        },
        {
          "bucket": "B",
          "color": 62,
          "colordes": "SWP本白 VG010M ",
          "colorsho": "SWP本白",
          "x": 3.08,
          "speed": 0.324675
        },
        {
          "bucket": "C",
          "color": 62,
          "colordes": "SWP本白 VG010M ",
          "colorsho": "SWP本白",
          "x": 1.0,
          "speed": 1.0
        },
        {
          "bucket": "D",
          "color": 81,
          "colordes": "W 白棉 VG055M ",
          "colorsho": "W 白棉",
          "x": 1.0,
          "speed": 1.0
        },
        {
          "bucket": "E",
          "color": 62,
          "colordes": "SWP本白 VG010M ",
          "colorsho": "SWP本白",
          "x": 1.13,
          "speed": 0.884956
        },
        {
          "bucket": "F",
          "color": 80,
          "colordes": "WJ 白棉 VG055M ",
          "colorsho": "WJ 白棉",
          "x": 1.13,
          "speed": 0.884956
        },
        {
          "bucket": "G",
          "color": 81,
          "colordes": "W 白棉 VG055M ",
          "colorsho": "W 白棉",
          "x": 3.08,
          "speed": 0.324675
        },
        {
          "bucket": "H",
          "color": 70,
          "colordes": "BC02W VE001M-U-001 ",
          "colorsho": "BC02W",
          "x": 1.49,
          "speed": 0.671141
        }
      ],
      "colors": [
        {
          "color": 62,
          "colordes": "BC02W VE001M-U-001 ",
          "colorsho": "BC02W",
          "target": 36.9,
          "final": 36.88,
          "error": "0.0194"
        },
        {
          "color": 70,
          "colordes": "BC02W VE001M-U-001 ",
          "colorsho": "BC02W",
          "target": 11.2,
          "final": 11.2,
          "error": "0.0019"
        },
        {
          "color": 80,
          "colordes": "BC02W VE001M-U-001 ",
          "colorsho": "BC02W",
          "target": 29.8,
          "final": 29.81,
          "error": "0.0075"
        },
        {
          "color": 81,
          "colordes": "BC02W VE001M-U-001 ",
          "colorsho": "BC02W",
          "target": 22.1,
          "final": 22.11,
          "error": "0.0100"
        }
      ]
    },
    {
      "X1": 1.11,
      "X2": 1.1,
      "X3": 1.47,
      "X4": 2.87,
      "cum_error": "0.000878",
      "total_feed_speed_D": 6.096219,
      "stage_label": "1.1-2.5/3.5",
      "assign": [
        {
          "bucket": "A",
          "color": 62,
          "colordes": "SWP本白 VG010M ",
          "colorsho": "SWP本白",
          "x": 1.11,
          "speed": 0.900901
        },
        {
          "bucket": "B",
          "color": 62,
          "colordes": "SWP本白 VG010M ",
          "colorsho": "SWP本白",
          "x": 2.87,
          "speed": 0.348432
        },
        {
          "bucket": "C",
          "color": 62,
          "colordes": "SWP本白 VG010M ",
          "colorsho": "SWP本白",
          "x": 1.0,
          "speed": 1.0
        },
        {
          "bucket": "D",
          "color": 81,
          "colordes": "W 白棉 VG055M ",
          "colorsho": "W 白棉",
          "x": 1.0,
          "speed": 1.0
        },
        {
          "bucket": "E",
          "color": 80,
          "colordes": "WJ 白棉 VG055M ",
          "colorsho": "WJ 白棉",
          "x": 1.1,
          "speed": 0.909091
        },
        {
          "bucket": "F",
          "color": 80,
          "colordes": "WJ 白棉 VG055M ",
          "colorsho": "WJ 白棉",
          "x": 1.1,
          "speed": 0.909091
        },
        {
          "bucket": "G",
          "color": 81,
          "colordes": "W 白棉 VG055M ",
          "colorsho": "W 白棉",
          "x": 2.87,
          "speed": 0.348432
        },
        {
          "bucket": "H",
          "color": 70,
          "colordes": "BC02W VE001M-U-001 ",
          "colorsho": "BC02W",
          "x": 1.47,
          "speed": 0.680272
        }
      ],
      "colors": [
        {
          "color": 62,
          "colordes": "BC02W VE001M-U-001 ",
          "colorsho": "BC02W",
          "target": 36.9,
          "final": 36.9,
          "error": "0.0028"
        },
        {
          "color": 70,
          "colordes": "BC02W VE001M-U-001 ",
          "colorsho": "BC02W",
          "target": 11.2,
          "final": 11.16,
          "error": "0.0411"
        },
        {
          "color": 80,
          "colordes": "BC02W VE001M-U-001 ",
          "colorsho": "BC02W",
          "target": 29.8,
          "final": 29.82,
          "error": "0.0247"
        },
        {
          "color": 81,
          "colordes": "BC02W VE001M-U-001 ",
          "colorsho": "BC02W",
          "target": 22.1,
          "final": 22.12,
          "error": "0.0192"
        }
      ]
    },
    {
      "X1": 1.13,
      "X2": 1.07,
      "X3": 1.33,
      "X4": 1.67,
      "cum_error": "0.001691",
      "total_feed_speed_D": 6.703599,
      "stage_label": "1.1-2.5/3.5",
      "assign": [
        {
          "bucket": "A",
          "color": 81,
          "colordes": "W 白棉 VG055M ",
          "colorsho": "W 白棉",
          "x": 1.13,
          "speed": 0.884956
        },
        {
          "bucket": "B",
          "color": 62,
          "colordes": "SWP本白 VG010M ",
          "colorsho": "SWP本白",
          "x": 1.67,
          "speed": 0.598802
        },
        {
          "bucket": "C",
          "color": 80,
          "colordes": "WJ 白棉 VG055M ",
          "colorsho": "WJ 白棉",
          "x": 1.0,
          "speed": 1.0
        },
        {
          "bucket": "D",
          "color": 80,
          "colordes": "WJ 白棉 VG055M ",
          "colorsho": "WJ 白棉",
          "x": 1.0,
          "speed": 1.0
        },
        {
          "bucket": "E",
          "color": 62,
          "colordes": "SWP本白 VG010M ",
          "colorsho": "SWP本白",
          "x": 1.07,
          "speed": 0.934579
        },
        {
          "bucket": "F",
          "color": 62,
          "colordes": "SWP本白 VG010M ",
          "colorsho": "SWP本白",
          "x": 1.07,
          "speed": 0.934579
        },
        {
          "bucket": "G",
          "color": 81,
          "colordes": "W 白棉 VG055M ",
          "colorsho": "W 白棉",
          "x": 1.67,
          "speed": 0.598802
        },
        {
          "bucket": "H",
          "color": 70,
          "colordes": "BC02W VE001M-U-001 ",
          "colorsho": "BC02W",
          "x": 1.33,
          "speed": 0.75188
        }
      ],
      "colors": [
        {
          "color": 62,
          "colordes": "BC02W VE001M-U-001 ",
          "colorsho": "BC02W",
          "target": 36.9,
          "final": 36.82,
          "error": "0.0845"
        },
        {
          "color": 70,
          "colordes": "BC02W VE001M-U-001 ",
          "colorsho": "BC02W",
          "target": 11.2,
          "final": 11.22,
          "error": "0.0161"
        },
        {
          "color": 80,
          "colordes": "BC02W VE001M-U-001 ",
          "colorsho": "BC02W",
          "target": 29.8,
          "final": 29.83,
          "error": "0.0347"
        },
        {
          "color": 81,
          "colordes": "BC02W VE001M-U-001 ",
          "colorsho": "BC02W",
          "target": 22.1,
          "final": 22.13,
          "error": "0.0338"
        }
      ]
    }
  ]
}';

-- 清理现有数据
DELETE
FROM MFGMATBTS
WHERE MFGNUM = @MFGNUM
  AND MFBLIN = @MFBLIN
  AND YARNGRP = @YARNGRP;

WITH BaseData
AS (
  SELECT @MFGNUM AS MFGNUM
    , @MFBLIN AS MFBLIN
    , @YARNGRP AS YARNGRP
    , results.[key] + 1 AS RESNUM
    , JSON_VALUE(results.value, '$.X1') AS X1
    , JSON_VALUE(results.value, '$.X2') AS X2
    , JSON_VALUE(results.value, '$.X3') AS X3
    , JSON_VALUE(results.value, '$.X4') AS X4
    , JSON_VALUE(results.value, '$.cum_error') AS cum_error
    , JSON_VALUE(results.value, '$.total_feed_speed_D') AS total_feed_speed_D
    , JSON_VALUE(results.value, '$.stage_label') AS stage_label
    , JSON_VALUE(assign.value, '$.bucket') AS bucket
    , JSON_VALUE(assign.value, '$.color') AS color
    , JSON_VALUE(assign.value, '$.colordes') AS colordes
    , JSON_VALUE(assign.value, '$.colorsho') AS colorsho
    , JSON_VALUE(assign.value, '$.x') AS x
    , JSON_VALUE(assign.value, '$.speed') AS speed
    , '#LOGIN_USER#' AS CREUSR
    , GETDATE() AS CRETIM
  FROM OPENJSON(@JSON, '$.results') AS results
  CROSS APPLY OPENJSON(results.value, '$.assign') AS assign
  )
-- INSERT INTO [MFGMATBTS] (
--   [MFGNUM]
--   , [MFBLIN]
--   , [YARNGRP]
--   , [RESNUM]
--   , [RESFLG]
--   , [STGLBL]
--   , [STRX]
--   , [CUMVAR]
--   , [FEEDSPEED]
--   , [MFMLINA]
--   , [MFMLINB]
--   , [MFMLINC]
--   , [MFMLIND]
--   , [MFMLINE]
--   , [MFMLINF]
--   , [MFMLING]
--   , [MFMLINH]
--   , [COLDESA]
--   , [COLDESB]
--   , [COLDESC]
--   , [COLDESD]
--   , [COLDESE]
--   , [COLDESF]
--   , [COLDESG]
--   , [COLDESH]
--   , [COLSHOA]
--   , [COLSHOB]
--   , [COLSHOC]
--   , [COLSHOD]
--   , [COLSHOE]
--   , [COLSHOF]
--   , [COLSHOG]
--   , [COLSHOH]
--   , [STRRATA]
--   , [STRRATB]
--   , [STRRATC]
--   , [STRRATD]
--   , [STRRATE]
--   , [STRRATF]
--   , [STRRATG]
--   , [STRRATH]
--   , [SPEEDA]
--   , [SPEEDB]
--   , [SPEEDC]
--   , [SPEEDD]
--   , [SPEEDE]
--   , [SPEEDF]
--   , [SPEEDG]
--   , [SPEEDH]
--   , [CREUSR]
--   , [CRETIM]
--   )
SELECT MFGNUM
  , MFBLIN
  , YARNGRP
  , RESNUM
  , 0 RESFLG
  , MAX(stage_label) AS STGLBL
  , MAX(CONCAT_WS('/', CAST(X1 AS decimal(18, 2)), CAST(X2 AS decimal(18, 2)), CAST(X3 AS decimal(18, 2)), CAST(X4 AS decimal(18, 2)))) AS STRX
  , MAX(cum_error) AS CUMVAR
  , MAX(total_feed_speed_D) AS FEEDSPEED
  -- Color 转置
  , MAX(CASE 
      WHEN bucket = 'A'
        THEN color
      END) AS MFMLINA
  , MAX(CASE 
      WHEN bucket = 'B'
        THEN color
      END) AS MFMLINB
  , MAX(CASE 
      WHEN bucket = 'C'
        THEN color
      END) AS MFMLINC
  , MAX(CASE 
      WHEN bucket = 'D'
        THEN color
      END) AS MFMLIND
  , MAX(CASE 
      WHEN bucket = 'E'
        THEN color
      END) AS MFMLINE
  , MAX(CASE 
      WHEN bucket = 'F'
        THEN color
      END) AS MFMLINF
  , MAX(CASE 
      WHEN bucket = 'G'
        THEN color
      END) AS MFMLING
  , MAX(CASE 
      WHEN bucket = 'H'
        THEN color
      END) AS MFMLINH
  -- Colordes 转置
  , MAX(CASE 
      WHEN bucket = 'A'
        THEN colordes
      END) AS COLDESA
  , MAX(CASE 
      WHEN bucket = 'B'
        THEN colordes
      END) AS COLDESB
  , MAX(CASE 
      WHEN bucket = 'C'
        THEN colordes
      END) AS COLDESC
  , MAX(CASE 
      WHEN bucket = 'D'
        THEN colordes
      END) AS COLDESD
  , MAX(CASE 
      WHEN bucket = 'E'
        THEN colordes
      END) AS COLDESE
  , MAX(CASE 
      WHEN bucket = 'F'
        THEN colordes
      END) AS COLDESF
  , MAX(CASE 
      WHEN bucket = 'G'
        THEN colordes
      END) AS COLDESG
  , MAX(CASE 
      WHEN bucket = 'H'
        THEN colordes
      END) AS COLDESH
  -- Colorsho 转置
  , MAX(CASE 
      WHEN bucket = 'A'
        THEN colorsho
      END) AS COLSHOA
  , MAX(CASE 
      WHEN bucket = 'B'
        THEN colorsho
      END) AS COLSHOB
  , MAX(CASE 
      WHEN bucket = 'C'
        THEN colorsho
      END) AS COLSHOC
  , MAX(CASE 
      WHEN bucket = 'D'
        THEN colorsho
      END) AS COLSHOD
  , MAX(CASE 
      WHEN bucket = 'E'
        THEN colorsho
      END) AS COLSHOE
  , MAX(CASE 
      WHEN bucket = 'F'
        THEN colorsho
      END) AS COLSHOF
  , MAX(CASE 
      WHEN bucket = 'G'
        THEN colorsho
      END) AS COLSHOG
  , MAX(CASE 
      WHEN bucket = 'H'
        THEN colorsho
      END) AS COLSHOH
  -- X 转置
  , MAX(CASE 
      WHEN bucket = 'A'
        THEN x
      END) AS STRRATA
  , MAX(CASE 
      WHEN bucket = 'B'
        THEN x
      END) AS STRRATB
  , MAX(CASE 
      WHEN bucket = 'C'
        THEN x
      END) AS STRRATC
  , MAX(CASE 
      WHEN bucket = 'D'
        THEN x
      END) AS STRRATD
  , MAX(CASE 
      WHEN bucket = 'E'
        THEN x
      END) AS STRRATE
  , MAX(CASE 
      WHEN bucket = 'F'
        THEN x
      END) AS STRRATF
  , MAX(CASE 
      WHEN bucket = 'G'
        THEN x
      END) AS STRRATG
  , MAX(CASE 
      WHEN bucket = 'H'
        THEN x
      END) AS STRRATH
  -- Speed 转置
  , MAX(CASE 
      WHEN bucket = 'A'
        THEN speed
      END) AS SPEEDA
  , MAX(CASE 
      WHEN bucket = 'B'
        THEN speed
      END) AS SPEEDB
  , MAX(CASE 
      WHEN bucket = 'C'
        THEN speed
      END) AS SPEEDC
  , MAX(CASE 
      WHEN bucket = 'D'
        THEN speed
      END) AS SPEEDD
  , MAX(CASE 
      WHEN bucket = 'E'
        THEN speed
      END) AS SPEEDE
  , MAX(CASE 
      WHEN bucket = 'F'
        THEN speed
      END) AS SPEEDF
  , MAX(CASE 
      WHEN bucket = 'G'
        THEN speed
      END) AS SPEEDG
  , MAX(CASE 
      WHEN bucket = 'H'
        THEN speed
      END) AS SPEEDH
  , CREUSR
  , CRETIM
FROM BaseData
GROUP BY MFGNUM
  , MFBLIN
  , YARNGRP
  , RESNUM
  , CREUSR
  , CRETIM
ORDER BY RESNUM;