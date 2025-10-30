-- 假设 JSON 文件内容已加载到变量 @json 中
DECLARE @json NVARCHAR(MAX);

-- 从文件读取 JSON 内容（实际使用时需替换为实际文件路径）
SET @json = 
  N'{
  "meta": {
    "targets_pct": [
      5.71,
      24.5,
      19.05,
      35.5,
      15.24
    ],
    "tol": 0.015,
    "refine_step": 0.01,
    "returned": 10
  },
  "results": [
    {
      "X1": 1.12,
      "X2": 1.97,
      "X3": 1.23,
      "X4": 3.28,
      "cum_error": 0.000555,
      "total_feed_speed_D": 5.33085,
      "assign": [
        {
          "bucket": "A",
          "color": 40,
          "x": 1.12,
          "speed": 0.892857
        },
        {
          "bucket": "B",
          "color": 10,
          "x": 3.28,
          "speed": 0.304878
        },
        {
          "bucket": "C",
          "color": 20,
          "x": 1.0,
          "speed": 1
        },
        {
          "bucket": "D",
          "color": 40,
          "x": 1.0,
          "speed": 1
        },
        {
          "bucket": "E",
          "color": 30,
          "x": 1.97,
          "speed": 0.507614
        },
        {
          "bucket": "F",
          "color": 30,
          "x": 1.97,
          "speed": 0.507614
        },
        {
          "bucket": "G",
          "color": 20,
          "x": 3.28,
          "speed": 0.304878
        },
        {
          "bucket": "H",
          "color": 50,
          "x": 1.23,
          "speed": 0.813008
        }
      ],
      "colors": [
        {
          "color": 10,
          "target": 5.71,
          "final": 5.72,
          "error": 0.01
        },
        {
          "color": 20,
          "target": 24.5,
          "final": 24.48,
          "error": 0.02
        },
        {
          "color": 30,
          "target": 19.05,
          "final": 19.04,
          "error": 0.01
        },
        {
          "color": 40,
          "target": 35.5,
          "final": 35.51,
          "error": 0.01
        },
        {
          "color": 50,
          "target": 15.24,
          "final": 15.25,
          "error": 0.01
        }
      ]
    },
    {
      "X1": 1.23,
      "X2": 1.97,
      "X3": 1.12,
      "X4": 3.28,
      "cum_error": 0.000555,
      "total_feed_speed_D": 5.33085,
      "assign": [
        {
          "bucket": "A",
          "color": 50,
          "x": 1.23,
          "speed": 0.813008
        },
        {
          "bucket": "B",
          "color": 10,
          "x": 3.28,
          "speed": 0.304878
        },
        {
          "bucket": "C",
          "color": 20,
          "x": 1.0,
          "speed": 1
        },
        {
          "bucket": "D",
          "color": 40,
          "x": 1.0,
          "speed": 1
        },
        {
          "bucket": "E",
          "color": 30,
          "x": 1.97,
          "speed": 0.507614
        },
        {
          "bucket": "F",
          "color": 30,
          "x": 1.97,
          "speed": 0.507614
        },
        {
          "bucket": "G",
          "color": 20,
          "x": 3.28,
          "speed": 0.304878
        },
        {
          "bucket": "H",
          "color": 40,
          "x": 1.12,
          "speed": 0.892857
        }
      ],
      "colors": [
        {
          "color": 10,
          "target": 5.71,
          "final": 5.72,
          "error": 0.01
        },
        {
          "color": 20,
          "target": 24.5,
          "final": 24.48,
          "error": 0.02
        },
        {
          "color": 30,
          "target": 19.05,
          "final": 19.04,
          "error": 0.01
        },
        {
          "color": 40,
          "target": 35.5,
          "final": 35.51,
          "error": 0.01
        },
        {
          "color": 50,
          "target": 15.24,
          "final": 15.25,
          "error": 0.01
        }
      ]
    },
    {
      "X1": 1.59,
      "X2": 1.33,
      "X3": 1.17,
      "X4": 3.11,
      "cum_error": 0.001202,
      "total_feed_speed_D": 5.630478,
      "assign": [
        {
          "bucket": "A",
          "color": 20,
          "x": 1.59,
          "speed": 0.628931
        },
        {
          "bucket": "B",
          "color": 10,
          "x": 3.11,
          "speed": 0.321543
        },
        {
          "bucket": "C",
          "color": 40,
          "x": 1.0,
          "speed": 1
        },
        {
          "bucket": "D",
          "color": 40,
          "x": 1.0,
          "speed": 1
        },
        {
          "bucket": "E",
          "color": 20,
          "x": 1.33,
          "speed": 0.75188
        },
        {
          "bucket": "F",
          "color": 30,
          "x": 1.33,
          "speed": 0.75188
        },
        {
          "bucket": "G",
          "color": 30,
          "x": 3.11,
          "speed": 0.321543
        },
        {
          "bucket": "H",
          "color": 50,
          "x": 1.17,
          "speed": 0.854701
        }
      ],
      "colors": [
        {
          "color": 10,
          "target": 5.71,
          "final": 5.71,
          "error": 0.0
        },
        {
          "color": 20,
          "target": 24.5,
          "final": 24.52,
          "error": 0.02
        },
        {
          "color": 30,
          "target": 19.05,
          "final": 19.06,
          "error": 0.01
        },
        {
          "color": 40,
          "target": 35.5,
          "final": 35.52,
          "error": 0.02
        },
        {
          "color": 50,
          "target": 15.24,
          "final": 15.18,
          "error": 0.06
        }
      ]
    },
    {
      "X1": 1.17,
      "X2": 1.33,
      "X3": 1.59,
      "X4": 3.11,
      "cum_error": 0.001202,
      "total_feed_speed_D": 5.630478,
      "assign": [
        {
          "bucket": "A",
          "color": 50,
          "x": 1.17,
          "speed": 0.854701
        },
        {
          "bucket": "B",
          "color": 10,
          "x": 3.11,
          "speed": 0.321543
        },
        {
          "bucket": "C",
          "color": 40,
          "x": 1.0,
          "speed": 1
        },
        {
          "bucket": "D",
          "color": 40,
          "x": 1.0,
          "speed": 1
        },
        {
          "bucket": "E",
          "color": 20,
          "x": 1.33,
          "speed": 0.75188
        },
        {
          "bucket": "F",
          "color": 30,
          "x": 1.33,
          "speed": 0.75188
        },
        {
          "bucket": "G",
          "color": 30,
          "x": 3.11,
          "speed": 0.321543
        },
        {
          "bucket": "H",
          "color": 20,
          "x": 1.59,
          "speed": 0.628931
        }
      ],
      "colors": [
        {
          "color": 10,
          "target": 5.71,
          "final": 5.71,
          "error": 0.0
        },
        {
          "color": 20,
          "target": 24.5,
          "final": 24.52,
          "error": 0.02
        },
        {
          "color": 30,
          "target": 19.05,
          "final": 19.06,
          "error": 0.01
        },
        {
          "color": 40,
          "target": 35.5,
          "final": 35.52,
          "error": 0.02
        },
        {
          "color": 50,
          "target": 15.24,
          "final": 15.18,
          "error": 0.06
        }
      ]
    },
    {
      "X1": 1.9,
      "X2": 1.17,
      "X3": 1.33,
      "X4": 3.11,
      "cum_error": 0.001213,
      "total_feed_speed_D": 5.630684,
      "assign": [
        {
          "bucket": "A",
          "color": 20,
          "x": 1.9,
          "speed": 0.526316
        },
        {
          "bucket": "B",
          "color": 10,
          "x": 3.11,
          "speed": 0.321543
        },
        {
          "bucket": "C",
          "color": 40,
          "x": 1.0,
          "speed": 1
        },
        {
          "bucket": "D",
          "color": 40,
          "x": 1.0,
          "speed": 1
        },
        {
          "bucket": "E",
          "color": 20,
          "x": 1.17,
          "speed": 0.854701
        },
        {
          "bucket": "F",
          "color": 50,
          "x": 1.17,
          "speed": 0.854701
        },
        {
          "bucket": "G",
          "color": 30,
          "x": 3.11,
          "speed": 0.321543
        },
        {
          "bucket": "H",
          "color": 30,
          "x": 1.33,
          "speed": 0.75188
        }
      ],
      "colors": [
        {
          "color": 10,
          "target": 5.71,
          "final": 5.71,
          "error": 0.0
        },
        {
          "color": 20,
          "target": 24.5,
          "final": 24.53,
          "error": 0.03
        },
        {
          "color": 30,
          "target": 19.05,
          "final": 19.06,
          "error": 0.01
        },
        {
          "color": 40,
          "target": 35.5,
          "final": 35.52,
          "error": 0.02
        },
        {
          "color": 50,
          "target": 15.24,
          "final": 15.18,
          "error": 0.06
        }
      ]
    },
    {
      "X1": 1.33,
      "X2": 1.17,
      "X3": 1.9,
      "X4": 3.11,
      "cum_error": 0.001213,
      "total_feed_speed_D": 5.630684,
      "assign": [
        {
          "bucket": "A",
          "color": 30,
          "x": 1.33,
          "speed": 0.75188
        },
        {
          "bucket": "B",
          "color": 10,
          "x": 3.11,
          "speed": 0.321543
        },
        {
          "bucket": "C",
          "color": 40,
          "x": 1.0,
          "speed": 1
        },
        {
          "bucket": "D",
          "color": 40,
          "x": 1.0,
          "speed": 1
        },
        {
          "bucket": "E",
          "color": 20,
          "x": 1.17,
          "speed": 0.854701
        },
        {
          "bucket": "F",
          "color": 50,
          "x": 1.17,
          "speed": 0.854701
        },
        {
          "bucket": "G",
          "color": 30,
          "x": 3.11,
          "speed": 0.321543
        },
        {
          "bucket": "H",
          "color": 20,
          "x": 1.9,
          "speed": 0.526316
        }
      ],
      "colors": [
        {
          "color": 10,
          "target": 5.71,
          "final": 5.71,
          "error": 0.0
        },
        {
          "color": 20,
          "target": 24.5,
          "final": 24.53,
          "error": 0.03
        },
        {
          "color": 30,
          "target": 19.05,
          "final": 19.06,
          "error": 0.01
        },
        {
          "color": 40,
          "target": 35.5,
          "final": 35.52,
          "error": 0.02
        },
        {
          "color": 50,
          "target": 15.24,
          "final": 15.18,
          "error": 0.06
        }
      ]
    },
    {
      "X1": 1.28,
      "X2": 2.0,
      "X3": 1.17,
      "X4": 3.37,
      "cum_error": 0.001467,
      "total_feed_speed_D": 5.229423,
      "assign": [
        {
          "bucket": "A",
          "color": 20,
          "x": 1.28,
          "speed": 0.78125
        },
        {
          "bucket": "B",
          "color": 10,
          "x": 3.37,
          "speed": 0.296736
        },
        {
          "bucket": "C",
          "color": 30,
          "x": 1.0,
          "speed": 1
        },
        {
          "bucket": "D",
          "color": 40,
          "x": 1.0,
          "speed": 1
        },
        {
          "bucket": "E",
          "color": 20,
          "x": 2.0,
          "speed": 0.5
        },
        {
          "bucket": "F",
          "color": 50,
          "x": 2.0,
          "speed": 0.5
        },
        {
          "bucket": "G",
          "color": 50,
          "x": 3.37,
          "speed": 0.296736
        },
        {
          "bucket": "H",
          "color": 40,
          "x": 1.17,
          "speed": 0.854701
        }
      ],
      "colors": [
        {
          "color": 10,
          "target": 5.71,
          "final": 5.67,
          "error": 0.04
        },
        {
          "color": 20,
          "target": 24.5,
          "final": 24.5,
          "error": 0.0
        },
        {
          "color": 30,
          "target": 19.05,
          "final": 19.12,
          "error": 0.07
        },
        {
          "color": 40,
          "target": 35.5,
          "final": 35.47,
          "error": 0.03
        },
        {
          "color": 50,
          "target": 15.24,
          "final": 15.24,
          "error": 0.0
        }
      ]
    },
    {
      "X1": 1.17,
      "X2": 2.0,
      "X3": 1.28,
      "X4": 3.37,
      "cum_error": 0.001467,
      "total_feed_speed_D": 5.229423,
      "assign": [
        {
          "bucket": "A",
          "color": 40,
          "x": 1.17,
          "speed": 0.854701
        },
        {
          "bucket": "B",
          "color": 10,
          "x": 3.37,
          "speed": 0.296736
        },
        {
          "bucket": "C",
          "color": 30,
          "x": 1.0,
          "speed": 1
        },
        {
          "bucket": "D",
          "color": 40,
          "x": 1.0,
          "speed": 1
        },
        {
          "bucket": "E",
          "color": 20,
          "x": 2.0,
          "speed": 0.5
        },
        {
          "bucket": "F",
          "color": 50,
          "x": 2.0,
          "speed": 0.5
        },
        {
          "bucket": "G",
          "color": 50,
          "x": 3.37,
          "speed": 0.296736
        },
        {
          "bucket": "H",
          "color": 20,
          "x": 1.28,
          "speed": 0.78125
        }
      ],
      "colors": [
        {
          "color": 10,
          "target": 5.71,
          "final": 5.67,
          "error": 0.04
        },
        {
          "color": 20,
          "target": 24.5,
          "final": 24.5,
          "error": 0.0
        },
        {
          "color": 30,
          "target": 19.05,
          "final": 19.12,
          "error": 0.07
        },
        {
          "color": 40,
          "target": 35.5,
          "final": 35.47,
          "error": 0.03
        },
        {
          "color": 50,
          "target": 15.24,
          "final": 15.24,
          "error": 0.0
        }
      ]
    },
    {
      "X1": 2.01,
      "X2": 1.17,
      "X3": 2.33,
      "X4": 3.33,
      "cum_error": 0.001752,
      "total_feed_speed_D": 5.236699,
      "assign": [
        {
          "bucket": "A",
          "color": 50,
          "x": 2.01,
          "speed": 0.497512
        },
        {
          "bucket": "B",
          "color": 10,
          "x": 3.33,
          "speed": 0.3003
        },
        {
          "bucket": "C",
          "color": 30,
          "x": 1.0,
          "speed": 1
        },
        {
          "bucket": "D",
          "color": 40,
          "x": 1.0,
          "speed": 1
        },
        {
          "bucket": "E",
          "color": 20,
          "x": 1.17,
          "speed": 0.854701
        },
        {
          "bucket": "F",
          "color": 40,
          "x": 1.17,
          "speed": 0.854701
        },
        {
          "bucket": "G",
          "color": 50,
          "x": 3.33,
          "speed": 0.3003
        },
        {
          "bucket": "H",
          "color": 20,
          "x": 2.33,
          "speed": 0.429185
        }
      ],
      "colors": [
        {
          "color": 10,
          "target": 5.71,
          "final": 5.73,
          "error": 0.02
        },
        {
          "color": 20,
          "target": 24.5,
          "final": 24.52,
          "error": 0.02
        },
        {
          "color": 30,
          "target": 19.05,
          "final": 19.1,
          "error": 0.05
        },
        {
          "color": 40,
          "target": 35.5,
          "final": 35.42,
          "error": 0.08
        },
        {
          "color": 50,
          "target": 15.24,
          "final": 15.24,
          "error": 0.0
        }
      ]
    },
    {
      "X1": 2.33,
      "X2": 1.17,
      "X3": 2.01,
      "X4": 3.33,
      "cum_error": 0.001752,
      "total_feed_speed_D": 5.236699,
      "assign": [
        {
          "bucket": "A",
          "color": 20,
          "x": 2.33,
          "speed": 0.429185
        },
        {
          "bucket": "B",
          "color": 10,
          "x": 3.33,
          "speed": 0.3003
        },
        {
          "bucket": "C",
          "color": 30,
          "x": 1.0,
          "speed": 1
        },
        {
          "bucket": "D",
          "color": 40,
          "x": 1.0,
          "speed": 1
        },
        {
          "bucket": "E",
          "color": 20,
          "x": 1.17,
          "speed": 0.854701
        },
        {
          "bucket": "F",
          "color": 40,
          "x": 1.17,
          "speed": 0.854701
        },
        {
          "bucket": "G",
          "color": 50,
          "x": 3.33,
          "speed": 0.3003
        },
        {
          "bucket": "H",
          "color": 50,
          "x": 2.01,
          "speed": 0.497512
        }
      ],
      "colors": [
        {
          "color": 10,
          "target": 5.71,
          "final": 5.73,
          "error": 0.02
        },
        {
          "color": 20,
          "target": 24.5,
          "final": 24.52,
          "error": 0.02
        },
        {
          "color": 30,
          "target": 19.05,
          "final": 19.1,
          "error": 0.05
        },
        {
          "color": 40,
          "target": 35.5,
          "final": 35.42,
          "error": 0.08
        },
        {
          "color": 50,
          "target": 15.24,
          "final": 15.24,
          "error": 0.0
        }
      ]
    }
  ]
}'
  ;

-- 方法1：直接提取所有 assign 数据
SELECT j1.[key] + 1 AS result
  , a.bucket
  , a.color
  , a.x
  , a.speed
FROM OPENJSON(@json, '$.results') AS j1
CROSS APPLY OPENJSON(j1.value, '$.assign') WITH (
    bucket VARCHAR(10)
    , color VARCHAR(30)
    , x DECIMAL(18, 2)
    , speed DECIMAL(18, 6)
    ) AS a;
  -- -- 方法2：关联 colors 数据（如果需要）
  -- SELECT j1.[key] AS result_index
  --   , a.bucket
  --   , a.color
  --   , c.target
  --   , c.final
  --   , c.error
  --   , a.x
  --   , a.speed
  -- FROM OPENJSON(@json, '$.results') AS j1
  -- CROSS APPLY OPENJSON(j1.value, '$.assign') WITH (
  --     bucket VARCHAR(10)
  --     , color INT
  --     , x FLOAT
  --     , speed FLOAT
  --     ) AS a
  -- OUTER APPLY (
  --   SELECT c.target
  --     , c.final
  --     , c.error
  --   FROM OPENJSON(j1.value, '$.colors') WITH (
  --       color INT '$.color'
  --       , target FLOAT '$.target'
  --       , final FLOAT '$.final'
  --       , error FLOAT '$.error'
  --       ) AS c
  --   WHERE c.color = a.color
  --   ) AS c;
