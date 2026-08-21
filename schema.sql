CREATE DATABASE IF NOT EXISTS harborstone_insurance;
USE harborstone_insurance;

CREATE TABLE Customers (
    customer_id INT AUTO_INCREMENT PRIMARY KEY,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    phone VARCHAR(20) NOT NULL,
    address VARCHAR(255),
    date_of_birth DATE,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE Employees (
    employee_id INT AUTO_INCREMENT PRIMARY KEY,
    full_name VARCHAR(100) NOT NULL,
    role VARCHAR(50) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    phone VARCHAR(20),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE Vessels (
    vessel_id INT AUTO_INCREMENT PRIMARY KEY,
    customer_id INT NOT NULL,
    vessel_name VARCHAR(100) NOT NULL,
    vessel_type VARCHAR(50),
    manufacturer VARCHAR(100),
    model VARCHAR(100),
    year_built INT,
    value DECIMAL(12,2),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES Customers(customer_id)
);

CREATE TABLE Policies (
    policy_id INT AUTO_INCREMENT PRIMARY KEY,
    customer_id INT NOT NULL,
    vessel_id INT NOT NULL,
    policy_type VARCHAR(100),
    start_date DATE,
    end_date DATE,
    premium DECIMAL(12,2),
    status VARCHAR(20),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (customer_id) REFERENCES Customers(customer_id),
    FOREIGN KEY (vessel_id) REFERENCES Vessels(vessel_id)
);

CREATE TABLE Claims (
    claim_id INT AUTO_INCREMENT PRIMARY KEY,
    policy_id INT NOT NULL,
    claim_date DATE,
    description TEXT,
    amount DECIMAL(12,2),
    status VARCHAR(20),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (policy_id) REFERENCES Policies(policy_id)
);

CREATE TABLE Payments (
    payment_id INT AUTO_INCREMENT PRIMARY KEY,
    policy_id INT NOT NULL,
    payment_date DATE,
    amount DECIMAL(12,2),
    payment_method VARCHAR(50),
    status VARCHAR(20),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (policy_id) REFERENCES Policies(policy_id)
);

-- ==========================================
-- State Graph Persistence & Failure Tickets
-- ==========================================

CREATE TABLE IF NOT EXISTS GraphRuns (
    run_id VARCHAR(100) PRIMARY KEY,
    graph_name VARCHAR(100) NOT NULL,

    customer_id INT NULL,
    policy_id INT NULL,
    claim_id INT NULL,
    vessel_id INT NULL,

    current_state VARCHAR(100) NOT NULL,
    status VARCHAR(50) NOT NULL,

    checkpoint_version INT NOT NULL DEFAULT 0,

    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        ON UPDATE CURRENT_TIMESTAMP,

    FOREIGN KEY (customer_id) REFERENCES Customers(customer_id),
    FOREIGN KEY (policy_id) REFERENCES Policies(policy_id),
    FOREIGN KEY (claim_id) REFERENCES Claims(claim_id),
    FOREIGN KEY (vessel_id) REFERENCES Vessels(vessel_id)
);

CREATE TABLE IF NOT EXISTS GraphCheckpoints (
    checkpoint_id INT AUTO_INCREMENT PRIMARY KEY,

    run_id VARCHAR(100) NOT NULL,
    checkpoint_version INT NOT NULL,

    current_state VARCHAR(100) NOT NULL,
    status VARCHAR(50) NOT NULL,

    state_json JSON NOT NULL,

    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (run_id)
        REFERENCES GraphRuns(run_id)
        ON DELETE CASCADE,

    UNIQUE KEY unique_run_checkpoint (
        run_id,
        checkpoint_version
    )
);

CREATE TABLE IF NOT EXISTS FailureTickets (
    ticket_id VARCHAR(100) PRIMARY KEY,
    run_id VARCHAR(100) NOT NULL,
    graph_name VARCHAR(100) NOT NULL,
    failed_node VARCHAR(100) NOT NULL,
    failure_type VARCHAR(50) NOT NULL,
    error_message TEXT NOT NULL,
    checkpoint_version INT NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'OPEN',
    recovery_attempts INT NOT NULL DEFAULT 0,
    resolution_note TEXT NULL,
    metadata JSON NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        ON UPDATE CURRENT_TIMESTAMP,

    FOREIGN KEY (run_id)
        REFERENCES GraphRuns(run_id)
        ON DELETE CASCADE
);

