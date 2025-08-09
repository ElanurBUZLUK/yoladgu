const API = {
  base: "http://localhost:8000/api/v1",

  async register(email, password, full_name, role="student"){
    const res = await fetch(`${this.base}/auth/register`, {method:"POST", headers:{'Content-Type':'application/json'}, body: JSON.stringify({email, password, full_name, role})});
    if(!res.ok) throw new Error("register failed");
    return res.json();
  },
  async login(email, password){
    const res = await fetch(`${this.base}/auth/login`, {method:"POST", headers:{'Content-Type':'application/json'}, body: JSON.stringify({email, password})});
    if(!res.ok) throw new Error("login failed");
    return res.json();
  },
  async vectorSearch(text, k=5){
    const res = await fetch(`${this.base}/vectors/search`, {method:"POST", headers:{'Content-Type':'application/json'}, body: JSON.stringify({text, k})});
    return res.json();
  },
  async teacherRequests(teacher_id){
    const res = await fetch(`${this.base}/teacher/students/requests?teacher_id=${teacher_id}`);
    return res.json();
  },
  async approveRequest(request_id, approve){
    const res = await fetch(`${this.base}/teacher/students/approve?request_id=${request_id}&approve=${approve}`, {method:"POST"});
    return res.json();
  },
  async listAssignmentsTeacher(teacher_id){
    const res = await fetch(`${this.base}/teacher/assignments?teacher_id=${teacher_id}`);
    return res.json();
  },
  async createAssignment(teacher_id, payload){
    const res = await fetch(`${this.base}/teacher/assignments?teacher_id=${teacher_id}`, {method:"POST", headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload)});
    return res.json();
  },
  async listAssignmentsStudent(student_id){
    const res = await fetch(`${this.base}/student/assignments?student_id=${student_id}`);
    return res.json();
  },
  async sendRequestTeacher(student_id, teacher_id){
    const res = await fetch(`${this.base}/student/request-teacher?student_id=${student_id}&teacher_id=${teacher_id}`, {method:"POST"});
    return res.json();
  }
};
