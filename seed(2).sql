USE harborstone_insurance;

INSERT INTO Customers (first_name, last_name, email, phone, address, date_of_birth)
VALUES
('John', 'Smith', 'john.smith@email.com', '555-1001', 'Miami, Florida', '1985-04-12'),
('Emma', 'Johnson', 'emma.johnson@email.com', '555-1002', 'Tampa, Florida', '1990-08-22'),
('Michael', 'Brown', 'michael.brown@email.com', '555-1003', 'Orlando, Florida', '1978-11-05'),
('Sophia', 'Davis', 'sophia.davis@email.com', '555-1004', 'Jacksonville, Florida', '1992-01-17'),
('William', 'Wilson', 'william.wilson@email.com', '555-1005', 'Naples, Florida', '1988-09-30'),
('Olivia', 'Taylor', 'olivia.taylor@email.com', '555-1006', 'Fort Myers, Florida', '1995-03-14'),
('James', 'Anderson', 'james.anderson@email.com', '555-1007', 'Key West, Florida', '1982-07-19'),
('Charlotte', 'Thomas', 'charlotte.thomas@email.com', '555-1008', 'Pensacola, Florida', '1993-12-08'),
('Benjamin', 'Moore', 'benjamin.moore@email.com', '555-1009', 'Sarasota, Florida', '1987-06-25'),
('Amelia', 'Martin', 'amelia.martin@email.com', '555-1010', 'Fort Lauderdale, Florida', '1996-10-11');

INSERT INTO Employees (full_name, role, email, phone)
VALUES
('David Miller', 'Claims Officer', 'david.miller@harborstone.com', '555-2001'),
('Sarah White', 'Claims Officer', 'sarah.white@harborstone.com', '555-2002'),
('Robert Harris', 'Insurance Agent', 'robert.harris@harborstone.com', '555-2003'),
('Jennifer Clark', 'Manager', 'jennifer.clark@harborstone.com', '555-2004'),
('Daniel Lewis', 'Customer Support', 'daniel.lewis@harborstone.com', '555-2005');

INSERT INTO Vessels (customer_id, vessel_name, vessel_type, manufacturer, model, year_built, value)
VALUES
(1,'Sea Breeze','Yacht','Sunseeker','Predator 60',2020,850000.00),
(2,'Ocean Star','Boat','Boston Whaler','280 Outrage',2019,180000.00),
(3,'Blue Horizon','Yacht','Azimut','55 Fly',2021,1200000.00),
(4,'Wave Rider','Boat','Sea Ray','SLX 310',2018,220000.00),
(5,'Coral Queen','Yacht','Princess','F50',2022,1350000.00),
(6,'Atlantic Dream','Boat','Regal','33 XO',2020,260000.00),
(7,'Silver Pearl','Yacht','Ferretti','670',2021,1800000.00),
(8,'Sea Explorer','Boat','Jeanneau','Leader 10.5',2019,210000.00),
(9,'Golden Tide','Yacht','Prestige','590',2023,1450000.00),
(10,'Ocean Spirit','Boat','Bayliner','VR6',2022,95000.00);

INSERT INTO Policies (customer_id,vessel_id,policy_type,start_date,end_date,premium,status) VALUES
(1,1,'Marine Hull Insurance','2025-01-01','2026-01-01',7500.00,'Active'),
(2,2,'Boat Insurance','2025-02-10','2026-02-10',2200.00,'Active'),
(3,3,'Yacht Insurance','2025-03-15','2026-03-15',9800.00,'Active'),
(4,4,'Boat Insurance','2025-04-20','2026-04-20',2500.00,'Active'),
(5,5,'Luxury Yacht Insurance','2025-05-05','2026-05-05',12000.00,'Active'),
(6,6,'Commercial Marine Insurance','2025-06-01','2026-06-01',4800.00,'Pending'),
(7,7,'Marine Hull Insurance','2025-07-18','2026-07-18',10500.00,'Active'),
(8,8,'Boat Insurance','2025-08-08','2026-08-08',2600.00,'Expired'),
(9,9,'Luxury Yacht Insurance','2025-09-12','2026-09-12',13500.00,'Active'),
(10,10,'Boat Insurance','2025-10-01','2026-10-01',1800.00,'Pending');

INSERT INTO Claims (policy_id,claim_date,description,amount,status) VALUES
(1,'2025-03-12','Hull damage after storm',12000.00,'Approved'),
(2,'2025-04-05','Engine failure',4500.00,'Pending'),
(3,'2025-05-18','Collision with another vessel',38000.00,'Approved'),
(4,'2025-06-10','Minor deck damage',2500.00,'Rejected'),
(5,'2025-06-28','Fire damage',75000.00,'Pending'),
(6,'2025-07-14','Propeller replacement',3200.00,'Approved'),
(7,'2025-08-09','Flood damage',18500.00,'Pending'),
(8,'2025-09-01','Broken navigation system',4100.00,'Approved'),
(9,'2025-09-20','Storm damage',27000.00,'Pending'),
(10,'2025-10-11','Dock collision',6500.00,'Approved');

INSERT INTO Payments (policy_id,payment_date,amount,payment_method,status) VALUES
(1,'2025-01-01',7500.00,'Credit Card','Paid'),
(2,'2025-02-10',2200.00,'Bank Transfer','Paid'),
(3,'2025-03-15',9800.00,'Credit Card','Paid'),
(4,'2025-04-20',2500.00,'Cash','Paid'),
(5,'2025-05-05',12000.00,'Bank Transfer','Paid'),
(6,'2025-06-01',4800.00,'Credit Card','Pending'),
(7,'2025-07-18',10500.00,'Credit Card','Paid'),
(8,'2025-08-08',2600.00,'Cash','Paid'),
(9,'2025-09-12',13500.00,'Bank Transfer','Pending'),
(10,'2025-10-01',1800.00,'Credit Card','Paid');
