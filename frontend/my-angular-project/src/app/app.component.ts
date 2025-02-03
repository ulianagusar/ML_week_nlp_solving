import { Component, OnInit } from '@angular/core';
import { HttpClientModule } from '@angular/common/http';
import { HttpClient } from '@angular/common/http';
import { CommonModule } from '@angular/common'; 

@Component({
  selector: 'app-root',
  standalone: true,  
  imports: [CommonModule, HttpClientModule],  
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {
  title = 'my-angular-project';
  posts: any[] = [];  
  filteredPosts: any[] = []; 
  model: any[] = [];  
  filteredModels: any[] = []; 
  channels: string[] = ['Вертолатте', 'ДРОННИЦА', 'Донбасс Россия']; 
  models: string[] = ['ruBert', 'xgboost'];
  selectedChannel: string = 'all';  
  selectedModels: string = 'all';
  selectedStartDate: string | undefined;
  selectedEndDate: string | undefined;

  constructor(private http: HttpClient) {}

  ngOnInit(): void {
    // Ініціалізація або виконання додаткових операцій, якщо потрібно
  }

  onChannelChange(event: any): void {
    this.selectedChannel = event.target.value;

    if (this.selectedChannel === 'all') {
      this.filteredPosts = this.posts; 
    } else {
      this.filteredPosts = this.posts.filter(post => post.Channel === this.selectedChannel);
    }
  }

  onModelsChange(event: any): void {
    this.selectedModels = event.target.value;

    if (this.selectedModels === 'all') {
      this.filteredModels = this.model; 
    } else {
      this.filteredModels = this.model.filter(post => post.models === this.selectedModels);
    }
  }

  onStartDateChange(event: any): void {
    this.selectedStartDate = event.target.value;
  }

  onEndDateChange(event: any): void {
    this.selectedEndDate = event.target.value;
  }

  onSearch(): void {
    const requestBody = {
      channel: this.selectedChannel,
      start_date: this.selectedStartDate,
      end_date: this.selectedEndDate,
      model: this.selectedModels
    };
    console.log(requestBody)
    // Надсилаємо POST запит для отримання повідомлень
    this.http.post('http://127.0.0.1:5000/api/fetch_posts', requestBody)
      .subscribe({
        next: () => {
          // Після успішного виконання POST запиту виконуємо GET запит для отримання повідомлень
          this.http.get<any[]>('http://127.0.0.1:5000/api/posts')
            .subscribe(data => {
              console.log(data);
              this.posts = data;
              this.filteredPosts = data;  // Оновлюємо список фільтрованих повідомлень
            }, error => {
              console.error('Помилка при отриманні постів:', error);
            });
        },
        error: (error) => {
          console.error('Помилка при виконанні POST запиту:', error);
        }
      });
  }
}
