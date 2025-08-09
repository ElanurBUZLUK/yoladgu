-- Users & Auth
create table if not exists users (
  id serial primary key,
  email text unique not null,
  password_hash text not null,
  full_name text,
  role text not null default 'student', -- 'admin' | 'teacher' | 'student'
  is_email_verified boolean not null default false,
  created_at timestamp without time zone default now()
);

create table if not exists refresh_tokens(
  id serial primary key,
  user_id int references users(id) on delete cascade,
  token text not null,
  expires_at timestamp without time zone not null,
  created_at timestamp without time zone default now()
);

-- Teacher-Student handshake (two-sided)
create table if not exists teacher_student_requests(
  id serial primary key,
  teacher_id int references users(id) on delete cascade,
  student_id int references users(id) on delete cascade,
  status text not null default 'pending', -- pending|approved|rejected|blocked
  created_at timestamp without time zone default now(),
  updated_at timestamp without time zone default now()
);

-- Assignments
create table if not exists assignments(
  id serial primary key,
  teacher_id int references users(id) on delete set null,
  title text not null,
  description text,
  due_date timestamp without time zone,
  topic text,
  created_at timestamp without time zone default now()
);

create table if not exists submissions(
  id serial primary key,
  assignment_id int references assignments(id) on delete cascade,
  student_id int references users(id) on delete cascade,
  content text,
  score numeric,
  created_at timestamp without time zone default now()
);

-- Simple analytics
create table if not exists events(
  id bigserial primary key,
  user_id int,
  event_type text,
  payload jsonb,
  created_at timestamp without time zone default now()
);

-- Indices
create index if not exists ix_events_user_id on events(user_id);
create index if not exists ix_tsr_teacher_student on teacher_student_requests(teacher_id, student_id);
