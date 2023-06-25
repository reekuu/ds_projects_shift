export PATH=/Library/PostgreSQL/14/bin:$PATH

echo All tables in home_credit will be dropped. Enter password to continue:
sudo -u postgres psql -f /Users/Kirill/GitHub/cft-modelling/notebooks/hw_sql/drop.sql home_credit

echo 7 tables in home_credit will be created. Enter password to continue:
sudo -u postgres psql -f /Users/Kirill/GitHub/cft-modelling/notebooks/hw_sql/create.sql home_credit

echo Tables in home_credit will be populated from CSV. Enter password to continue:
sudo -u postgres psql -f /Users/Kirill/GitHub/cft-modelling/notebooks/hw_sql/copy.sql home_credit