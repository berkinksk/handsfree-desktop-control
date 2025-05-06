CREATE TABLE IF NOT EXISTS settings   (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS calibration(key TEXT PRIMARY KEY, value REAL);
CREATE TABLE IF NOT EXISTS metrics    (session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                                       start_time TEXT, duration REAL, clicks INTEGER); 